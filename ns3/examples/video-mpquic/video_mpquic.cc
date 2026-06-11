/*
 * video_mpquic.cc — NS-3 scenario for RL-controlled adaptive video over
 * (abstracted) multipath QUIC, driven from Python via the ns3-ai shared-memory
 * message interface (struct-based).
 *
 * Role: this is the C++ "body" that the Python "brain" (src/ns3env + src/rl)
 * drives. It is intentionally THIN — all RL logic, observation assembly and
 * reward live in Python. C++ only:
 *   1) builds an N-path topology (multi-homed client <-> video server), each
 *      path a stock single-path TCP subflow with its own bottleneck + queue +
 *      time-varying UDP cross-traffic (abstracted multipath);
 *   2) reports per-path observable stats each decision epoch;
 *   3) applies the agent's action (which subflow to fetch the next segment on)
 *      and reports the realized download.
 *
 * Decision epoch = ONE video segment download. The struct fields (video_mpquic.h)
 * mirror the Python dataclasses PathStats / DownloadResult so the Python
 * Ns3DataPlane just marshals them across the bridge.
 *
 * Protocol (matches src/ns3env/dataplane.py::Ns3DataPlane):
 *   C++ leads with a send. Per decision the controller:
 *     FillObservation(env); CppSend(env);
 *     if env.done: Simulator::Stop(); return;   // terminal obs, no recv
 *     CppRecv(act); StartSegment(act);           // download runs asynchronously
 *   When the segment completes, OnSegmentDone records the realized result into
 *   env.last* and schedules the next decision.
 */

#include "video_mpquic.h"

#include <ns3/ai-module.h>
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("VideoMpquic");

// --------------------------------------------------------------------------- //
// SegmentSource — server-side re-armable bulk sender over a persistent TCP
// connection. StartSegment(bytes) pushes exactly `bytes` to the paired sink,
// draining the socket's send buffer as space frees (like BulkSendApplication).
// Also maintains a smoothed RTT estimate from the socket's "RTT" trace.
// --------------------------------------------------------------------------- //

class SegmentSource : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("VideoMpquic::SegmentSource")
                                .SetParent<Application>()
                                .SetGroupName("Applications")
                                .AddConstructor<SegmentSource>();
        return tid;
    }

    void Configure(Address peer, uint32_t pathIdx)
    {
        m_peer = peer;
        m_pathIdx = pathIdx;
    }

    // Begin delivering `bytes` application bytes to the sink. Idempotent w.r.t.
    // connection state: if the socket is not yet connected the data is sent as
    // soon as ConnectSucceeded fires.
    void StartSegment(uint32_t bytes)
    {
        m_remaining = bytes;
        if (m_connected)
        {
            SendData();
        }
    }

    double GetRttMs() const { return m_rttEwmaMs; }

  private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_socket->Bind();
        m_socket->Connect(m_peer);
        m_socket->SetConnectCallback(MakeCallback(&SegmentSource::ConnectSucceeded, this),
                                     MakeCallback(&SegmentSource::ConnectFailed, this));
        m_socket->SetSendCallback(MakeCallback(&SegmentSource::OnSendPossible, this));
        // The "RTT" trace exists on TcpSocketBase; connect it once the object exists.
        m_socket->TraceConnectWithoutContext("RTT", MakeCallback(&SegmentSource::RttTrace, this));
    }

    void StopApplication() override
    {
        if (m_socket)
        {
            m_socket->Close();
            m_socket = nullptr;
        }
    }

    void ConnectSucceeded(Ptr<Socket>)
    {
        m_connected = true;
        if (m_remaining > 0)
        {
            SendData();
        }
    }

    void ConnectFailed(Ptr<Socket> s)
    {
        NS_LOG_WARN("SegmentSource path " << m_pathIdx << " connect failed");
    }

    // Push as much of the remaining segment as the send buffer allows.
    void SendData()
    {
        while (m_remaining > 0)
        {
            uint32_t avail = m_socket->GetTxAvailable();
            if (avail == 0)
            {
                break; // wait for OnSendPossible
            }
            uint32_t toSend = static_cast<uint32_t>(std::min<uint64_t>(m_remaining, avail));
            int sent = m_socket->Send(Create<Packet>(toSend));
            if (sent <= 0)
            {
                break;
            }
            m_remaining -= static_cast<uint64_t>(sent);
        }
    }

    void OnSendPossible(Ptr<Socket>, uint32_t)
    {
        if (m_remaining > 0)
        {
            SendData();
        }
    }

    void RttTrace(Time, Time newRtt)
    {
        double sample = newRtt.GetSeconds() * 1000.0;
        if (m_rttEwmaMs <= 0.0)
        {
            m_rttEwmaMs = sample;
        }
        else
        {
            m_rttEwmaMs = 0.85 * m_rttEwmaMs + 0.15 * sample;
        }
    }

    Ptr<Socket> m_socket;
    Address m_peer;
    uint32_t m_pathIdx = 0;
    uint64_t m_remaining = 0;
    bool m_connected = false;
    double m_rttEwmaMs = 0.0;
};

// --------------------------------------------------------------------------- //
// SegmentSink — client-side receiver. ArmSegment(target) starts counting
// delivered bytes for one segment; when `target` bytes have arrived it reports
// (pathIdx, bytes, duration) via the completion callback.
// --------------------------------------------------------------------------- //

class SegmentSink : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("VideoMpquic::SegmentSink")
                                .SetParent<Application>()
                                .SetGroupName("Applications")
                                .AddConstructor<SegmentSink>();
        return tid;
    }

    void Configure(Address bindAddr,
                   uint32_t pathIdx,
                   Callback<void, uint32_t, uint32_t, Time> onDone)
    {
        m_bind = bindAddr;
        m_pathIdx = pathIdx;
        m_onDone = onDone;
    }

    void ArmSegment(uint32_t target)
    {
        m_target = target;
        m_segRx = 0;
        m_active = true;
        m_segStart = Simulator::Now();
    }

  private:
    void StartApplication() override
    {
        m_listen = Socket::CreateSocket(GetNode(), TcpSocketFactory::GetTypeId());
        m_listen->Bind(m_bind);
        m_listen->Listen();
        m_listen->SetAcceptCallback(
            MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
            MakeCallback(&SegmentSink::HandleAccept, this));
    }

    void StopApplication() override
    {
        if (m_listen)
        {
            m_listen->Close();
            m_listen = nullptr;
        }
    }

    void HandleAccept(Ptr<Socket> s, const Address&)
    {
        s->SetRecvCallback(MakeCallback(&SegmentSink::HandleRead, this));
    }

    void HandleRead(Ptr<Socket> s)
    {
        Ptr<Packet> pkt;
        Address from;
        while ((pkt = s->RecvFrom(from)))
        {
            uint32_t n = pkt->GetSize();
            if (n == 0)
            {
                break;
            }
            if (m_active)
            {
                m_segRx += n;
                if (m_segRx >= m_target)
                {
                    m_active = false;
                    Time dur = Simulator::Now() - m_segStart;
                    m_onDone(m_pathIdx, static_cast<uint32_t>(m_segRx), dur);
                }
            }
        }
    }

    Ptr<Socket> m_listen;
    Address m_bind;
    uint32_t m_pathIdx = 0;
    uint64_t m_target = 0;
    uint64_t m_segRx = 0;
    bool m_active = false;
    Time m_segStart;
    Callback<void, uint32_t, uint32_t, Time> m_onDone;
};

// --------------------------------------------------------------------------- //
// Scenario configuration
// --------------------------------------------------------------------------- //

struct PathLink
{
    std::string rate;  // bottleneck data rate, e.g. "8Mbps"
    std::string delay; // one-way propagation delay, e.g. "10ms"
    double crossFrac;  // mean cross-traffic as a fraction of `rate`
};

struct ScenarioConfig
{
    // Asymmetric 3-path access (wired / Wi-Fi / LTE-like), mirroring the mock
    // data plane's default so behaviour is comparable across backends.
    std::vector<PathLink> paths = {
        {"8Mbps", "10ms", 0.45},
        {"4Mbps", "17ms", 0.65},
        {"2Mbps", "30ms", 0.35},
    };
    uint32_t episodeSegments = 48;
    uint32_t segmentBytes = 500000;
    uint32_t seed = 1;
    uint16_t basePort = 5000;
};

// --------------------------------------------------------------------------- //
// VideoController — owns the topology and drives the per-segment decision loop
// over the ns3-ai message interface.
// --------------------------------------------------------------------------- //

class VideoController
{
  public:
    explicit VideoController(const ScenarioConfig& cfg) : m_cfg(cfg) {}

    void Build()
    {
        const uint32_t n = NumPaths();
        NS_ABORT_MSG_IF(n == 0 || n > kMaxPaths, "path count out of range");

        m_client.Create(1);
        m_server.Create(1);
        InternetStackHelper internet;
        internet.Install(m_client);
        internet.Install(m_server);

        m_throughputEwma.assign(n, 0.0);
        m_lossEwma.assign(n, 0.0);
        m_sources.resize(n);
        m_sinks.resize(n);
        m_clientAddr.resize(n);

        Ipv4AddressHelper addr;
        for (uint32_t i = 0; i < n; ++i)
        {
            PointToPointHelper p2p;
            p2p.SetDeviceAttribute("DataRate", StringValue(m_cfg.paths[i].rate));
            p2p.SetChannelAttribute("Delay", StringValue(m_cfg.paths[i].delay));
            NetDeviceContainer dev = p2p.Install(m_client.Get(0), m_server.Get(0));

            // Per-path AQM so loss + queueing delay emerge under load.
            TrafficControlHelper tch;
            tch.SetRootQueueDisc("ns3::FqCoDelQueueDisc");
            tch.Install(dev);

            std::ostringstream net;
            net << "10.1." << (i + 1) << ".0";
            addr.SetBase(net.str().c_str(), "255.255.255.0");
            Ipv4InterfaceContainer ifc = addr.Assign(dev);
            // dev.Get(0) is the client side, dev.Get(1) the server side.
            Ipv4Address clientIp = ifc.GetAddress(0);
            m_clientAddr[i] = clientIp;

            // Nominal link rate (Mbps) seeds the throughput estimate so idle
            // paths start at capacity (a real client's stale subflow estimate).
            DataRate dr(m_cfg.paths[i].rate);
            m_throughputEwma[i] = dr.GetBitRate() / 1e6;

            BuildPathApps(i, clientIp);
            BuildCrossTraffic(i, clientIp, dr);
        }

        Ipv4GlobalRoutingHelper::PopulateRoutingTables();

        // m_fmh is a member: FlowMonitorHelper's destructor disposes the monitor
        // (nulling each probe's Ipv4), so it must outlive Simulator::Run().
        m_monitor = m_fmh.InstallAll();
        m_classifier = DynamicCast<Ipv4FlowClassifier>(m_fmh.GetClassifier());
    }

    void AttachInterface(Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>* msg) { m_msg = msg; }

    void SetSelftest(bool on) { m_selftest = on; }

    // Kick off the loop once connections have had time to establish.
    void Start(Time warmup) { Simulator::Schedule(warmup, &VideoController::Decide, this); }

    uint32_t NumPaths() const { return static_cast<uint32_t>(m_cfg.paths.size()); }

  private:
    void BuildPathApps(uint32_t i, Ipv4Address clientIp)
    {
        uint16_t port = m_cfg.basePort + i;

        Ptr<SegmentSink> sink = CreateObject<SegmentSink>();
        sink->Configure(InetSocketAddress(Ipv4Address::GetAny(), port),
                        i,
                        MakeCallback(&VideoController::OnSegmentDone, this));
        m_client.Get(0)->AddApplication(sink);
        sink->SetStartTime(Seconds(0.0));
        m_sinks[i] = sink;

        Ptr<SegmentSource> src = CreateObject<SegmentSource>();
        src->Configure(InetSocketAddress(clientIp, port), i);
        m_server.Get(0)->AddApplication(src);
        src->SetStartTime(Seconds(0.1)); // after the sink is listening
        m_sources[i] = src;
    }

    // Time-varying UDP cross-traffic competing with the segment flow on path i.
    void BuildCrossTraffic(uint32_t i, Ipv4Address clientIp, DataRate linkRate)
    {
        uint16_t port = m_cfg.basePort + 100 + i;

        PacketSinkHelper csink("ns3::UdpSocketFactory",
                               InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer ca = csink.Install(m_client.Get(0));
        ca.Start(Seconds(0.0));

        double crossBps = m_cfg.paths[i].crossFrac * linkRate.GetBitRate();
        OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(clientIp, port));
        onoff.SetAttribute("DataRate", DataRateValue(DataRate(static_cast<uint64_t>(crossBps))));
        onoff.SetAttribute("PacketSize", UintegerValue(1200));
        // Bursty on/off so available bandwidth varies over time; phase-shift per
        // path via different mean on/off so paths peak at different times.
        std::ostringstream on, off;
        on << "ns3::ExponentialRandomVariable[Mean=" << (0.6 + 0.2 * i) << "]";
        off << "ns3::ExponentialRandomVariable[Mean=" << (0.8 + 0.3 * i) << "]";
        onoff.SetAttribute("OnTime", StringValue(on.str()));
        onoff.SetAttribute("OffTime", StringValue(off.str()));
        ApplicationContainer co = onoff.Install(m_server.Get(0));
        co.Start(Seconds(0.2));
        m_crossApps.Add(co);
    }

    // -- decision loop ------------------------------------------------------ //

    // One decision = one send (observation) + one receive (action). The
    // simulation is continuing: episode boundaries are reset bookkeeping, not a
    // process restart, so the same NS-3 process serves the whole training run.
    void Decide()
    {
        EnvStruct env{};
        FillObservation(env);
        env.done = (m_segmentsDone >= m_cfg.episodeSegments) ? 1 : 0;

        ActStruct act{};
        if (m_selftest)
        {
            // No bridge: round-robin policy in C++, terminate after one episode.
            if (env.done)
            {
                Simulator::Stop();
                return;
            }
            act.command = ACT_STEP;
            act.pathIdx = static_cast<int32_t>(m_segmentsDone % NumPaths());
            act.segmentBytes = m_cfg.segmentBytes;
        }
        else
        {
            m_msg->CppSendBegin();
            *m_msg->GetCpp2PyStruct() = env;
            m_msg->CppSendEnd();

            m_msg->CppRecvBegin();
            act = *m_msg->GetPy2CppStruct();
            m_msg->CppRecvEnd();
        }

        if (act.command == ACT_TERMINATE)
        {
            Simulator::Stop();
            return;
        }
        if (act.command == ACT_RESET || env.done)
        {
            // Start a new episode: reset counters but keep the network running
            // (warm subflows + evolving congestion). A non-reset action while
            // done is treated leniently as a reset.
            m_segmentsDone = 0;
            m_last.valid = false;
            Simulator::ScheduleNow(&VideoController::Decide, this);
            return;
        }

        int idx = act.pathIdx;
        NS_ABORT_MSG_IF(idx < 0 || idx >= static_cast<int>(NumPaths()), "pathIdx out of range");
        uint32_t bytes = act.segmentBytes ? act.segmentBytes : m_cfg.segmentBytes;

        m_curPath = static_cast<uint32_t>(idx);
        m_curBytes = bytes;
        m_sinks[idx]->ArmSegment(bytes);
        m_sources[idx]->StartSegment(bytes);
        // OnSegmentDone fires when the sink has received all `bytes`.
    }

    void OnSegmentDone(uint32_t pathIdx, uint32_t bytes, Time duration)
    {
        double dur = std::max(duration.GetSeconds(), 1e-6);
        double goodputMbps = (static_cast<double>(bytes) * 8.0) / (dur * 1e6);
        double rttMs = m_sources[pathIdx]->GetRttMs();
        double loss = SampleLoss(pathIdx);

        // EWMA the per-path throughput/loss estimates used in observations.
        m_throughputEwma[pathIdx] = 0.5 * m_throughputEwma[pathIdx] + 0.5 * goodputMbps;
        m_lossEwma[pathIdx] = 0.7 * m_lossEwma[pathIdx] + 0.3 * loss;

        m_last.path = static_cast<int32_t>(pathIdx);
        m_last.throughputMbps = goodputMbps;
        m_last.rttMs = rttMs;
        m_last.loss = loss;
        m_last.durationS = dur;
        m_last.bytes = bytes;
        m_last.valid = true;

        ++m_segmentsDone;
        Simulator::ScheduleNow(&VideoController::Decide, this);
    }

    void FillObservation(EnvStruct& env)
    {
        const uint32_t n = NumPaths();
        env.numPaths = n;
        env.clockS = Simulator::Now().GetSeconds();
        for (uint32_t i = 0; i < n; ++i)
        {
            env.throughputMbps[i] = m_throughputEwma[i];
            double rtt = m_sources[i]->GetRttMs();
            env.rttMs[i] = rtt > 0.0 ? rtt : 0.0;
            env.loss[i] = m_lossEwma[i];
        }
        if (m_last.valid)
        {
            env.lastPath = m_last.path;
            env.lastThroughputMbps = m_last.throughputMbps;
            env.lastRttMs = m_last.rttMs;
            env.lastLoss = m_last.loss;
            env.lastDurationS = m_last.durationS;
            env.lastBytes = m_last.bytes;
        }
        else
        {
            env.lastPath = -1;
        }
    }

    // Per-path loss from FlowMonitor: lost / (tx + lost) for the server->client
    // segment flow on path i, computed as a delta since the last observation so
    // it reflects recent conditions rather than the whole episode.
    double SampleLoss(uint32_t pathIdx)
    {
        if (!m_monitor || !m_classifier)
        {
            return 0.0;
        }
        m_monitor->CheckForLostPackets();
        auto stats = m_monitor->GetFlowStats();
        for (const auto& kv : stats)
        {
            Ipv4FlowClassifier::FiveTuple t = m_classifier->FindFlow(kv.first);
            // Segment flow: TCP toward this path's client address.
            if (t.destinationAddress == m_clientAddr[pathIdx] && t.protocol == 6)
            {
                uint64_t tx = kv.second.txPackets;
                uint64_t lost = kv.second.lostPackets;
                uint64_t denom = tx + lost;
                if (denom == 0)
                {
                    return 0.0;
                }
                return std::min(1.0, static_cast<double>(lost) / static_cast<double>(denom));
            }
        }
        return 0.0;
    }

    struct LastResult
    {
        bool valid = false;
        int32_t path = -1;
        double throughputMbps = 0.0;
        double rttMs = 0.0;
        double loss = 0.0;
        double durationS = 0.0;
        uint32_t bytes = 0;
    };

    ScenarioConfig m_cfg;
    NodeContainer m_client;
    NodeContainer m_server;
    std::vector<Ptr<SegmentSource>> m_sources;
    std::vector<Ptr<SegmentSink>> m_sinks;
    std::vector<Ipv4Address> m_clientAddr;
    ApplicationContainer m_crossApps;
    std::vector<double> m_throughputEwma;
    std::vector<double> m_lossEwma;
    FlowMonitorHelper m_fmh;
    Ptr<FlowMonitor> m_monitor;
    Ptr<Ipv4FlowClassifier> m_classifier;
    Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>* m_msg = nullptr;
    LastResult m_last;
    uint32_t m_segmentsDone = 0;
    uint32_t m_curPath = 0;
    uint32_t m_curBytes = 0;
    bool m_selftest = false;
};

// --------------------------------------------------------------------------- //

int
main(int argc, char* argv[])
{
    ScenarioConfig cfg;
    uint32_t segments = cfg.episodeSegments;
    uint32_t segmentBytes = cfg.segmentBytes;
    uint32_t seed = cfg.seed;
    bool selftest = false;

    CommandLine cmd;
    cmd.AddValue("segments", "Segments per episode", segments);
    cmd.AddValue("segmentBytes", "Bytes per segment", segmentBytes);
    cmd.AddValue("seed", "RNG seed", seed);
    cmd.AddValue("selftest", "Run a self-contained RR episode without the bridge", selftest);
    cmd.Parse(argc, argv);

    cfg.episodeSegments = segments;
    cfg.segmentBytes = segmentBytes;
    cfg.seed = seed;

    RngSeedManager::SetSeed(cfg.seed);

    VideoController controller(cfg);
    controller.Build();
    controller.SetSelftest(selftest);

    if (!selftest)
    {
        // ns3-ai struct-based message interface. Python is the memory creator,
        // so C++ passes isCreator=false. handleFinish=true makes the destructor
        // signal Python when the process ends.
        Ns3AiMsgInterface* interface = Ns3AiMsgInterface::Get();
        interface->SetIsMemoryCreator(false);
        interface->SetUseVector(false);
        interface->SetHandleFinish(true);
        Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>* msg =
            interface->GetInterface<EnvStruct, ActStruct>();
        controller.AttachInterface(msg);
    }

    // Warm up: let the per-path TCP connections establish before the first
    // decision so RTT estimates and sockets are live.
    controller.Start(Seconds(1.0));

    // Safety stop in case the bridge desyncs. The process normally runs until
    // Python sends ACT_TERMINATE (or is killed); the simulation is continuing
    // across episodes, so this bound is deliberately large.
    Simulator::Stop(Seconds(1e9));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
