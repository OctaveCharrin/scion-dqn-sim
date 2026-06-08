/*
 * video_mpquic.cc — NS-3 scenario for RL-controlled adaptive video over
 * (abstracted) multipath QUIC.  [SKELETON — build in WSL2, see ns3/README.md]
 *
 * Role: this is the C++ "body" that the Python "brain" (src/ns3env + src/rl)
 * drives over the ns3-ai shared-memory bridge. It is intentionally THIN — all
 * RL logic, observation assembly and reward live in Python. C++ only:
 *   1) builds an N-path topology (multi-homed client -> video server),
 *   2) reports per-path network stats each decision epoch,
 *   3) applies the agent's action (Phase 1: which path to fetch the next
 *      segment on) and reports the realized download.
 *
 * Decision epoch = ONE video segment download. The struct fields below mirror
 * the Python dataclasses in src/ns3env/dataplane.py (PathStats / DownloadResult)
 * so the Ns3DataPlane just marshals these across the bridge.
 *
 * This file does not compile as-is: the ns3-ai message-interface includes and a
 * few app hooks are marked `TODO(wsl2)`. Wire them against the pinned ns3-ai
 * version (ns3/requirements-ns3.md) once the NS-3 tree is set up.
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

// TODO(wsl2): include the ns3-ai message interface, e.g.
//   #include <ns3/ai-module.h>            // Ns3AiMsgInterface
// and define the struct-pair below as the shared-memory payload.

#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("VideoMpquic");

// ---- Shared-memory payload (mirrors src/ns3env/dataplane.py) --------------- //

static constexpr uint32_t kMaxPaths = 8;

// C++ -> Python: observable per-path state at a decision point.
struct EnvStruct
{
  uint32_t numPaths;
  double throughputMbps[kMaxPaths]; // EWMA goodput estimate per path
  double rttMs[kMaxPaths];          // smoothed RTT per path
  double loss[kMaxPaths];           // loss estimate per path
  double clockS;                    // current sim time
  uint8_t done;                     // 1 once the episode horizon is reached
  // Set after a download completes (mirrors DownloadResult):
  int32_t lastPath;                 // -1 before the first download
  double lastThroughputMbps;
  double lastRttMs;
  double lastLoss;
  double lastDurationS;
};

// Python -> C++: the agent's action for the next segment.
struct ActStruct
{
  int32_t pathIdx;       // Phase 1: which subflow to fetch on
  uint32_t segmentBytes; // segment size to deliver this epoch
  // Phase 2+: double sendRateMbps;  uint32_t bitrateLevel;  (extend here)
};

// ---- Scenario configuration ----------------------------------------------- //

struct PathLink
{
  std::string rate;  // e.g. "8Mbps"
  std::string delay; // e.g. "10ms"
};

struct ScenarioConfig
{
  // Default: asymmetric 3-path access (wired / Wi-Fi / LTE-like).
  std::vector<PathLink> paths = {
      {"8Mbps", "10ms"},
      {"4Mbps", "17ms"},
      {"2Mbps", "30ms"},
  };
  uint32_t episodeSegments = 48;
  uint32_t segmentBytes = 500000;
  uint32_t crossTrafficSeed = 1;
};

// ---- Topology -------------------------------------------------------------- //
//
// One client node and one server node connected by N parallel point-to-point
// links (one per "path"), each with its own bottleneck rate/delay, queue, and
// an OnOff cross-traffic source to make congestion time-varying. Each path runs
// an independent transport subflow (abstracted multipath): in Phase 1 the agent
// picks which subflow carries the next segment.

class VideoMpquicScenario
{
public:
  explicit VideoMpquicScenario(const ScenarioConfig& cfg) : m_cfg(cfg) {}

  void Build()
  {
    m_client.Create(1);
    m_server.Create(1);
    InternetStackHelper internet;
    internet.Install(m_client);
    internet.Install(m_server);

    const uint32_t n = static_cast<uint32_t>(m_cfg.paths.size());
    NS_ABORT_MSG_IF(n == 0 || n > kMaxPaths, "path count out of range");

    Ipv4AddressHelper addr;
    for (uint32_t i = 0; i < n; ++i)
      {
        PointToPointHelper p2p;
        p2p.SetDeviceAttribute("DataRate", StringValue(m_cfg.paths[i].rate));
        p2p.SetChannelAttribute("Delay", StringValue(m_cfg.paths[i].delay));
        NetDeviceContainer dev = p2p.Install(m_client.Get(0), m_server.Get(0));

        // Per-path queue/AQM so loss + queueing delay emerge under load.
        TrafficControlHelper tch;
        tch.SetRootQueueDisc("ns3::FqCoDelQueueDisc");
        tch.Install(dev);

        std::ostringstream net;
        net << "10.1." << (i + 1) << ".0";
        addr.SetBase(net.str().c_str(), "255.255.255.0");
        m_ifaces.push_back(addr.Assign(dev));

        // TODO(wsl2): install one transport subflow per path here
        //   - Phase 1–3 (abstracted): a stock single-path QUIC (or TCP) socket
        //     bound to this path's client/server addresses.
        //   - Phase 4 (true MPQUIC): a single MPQUIC connection whose scheduler
        //     the agent replaces/tunes.
        // TODO(wsl2): install an OnOff cross-traffic app on this path, seeded
        //   from m_cfg.crossTrafficSeed + i, to drive time-varying congestion.
      }
  }

  // Measure observable per-path stats for the next EnvStruct. Replace the
  // placeholders with real estimates from the subflow sockets / flow monitor.
  void FillObservation(EnvStruct& env) const
  {
    const uint32_t n = static_cast<uint32_t>(m_cfg.paths.size());
    env.numPaths = n;
    env.clockS = Simulator::Now().GetSeconds();
    for (uint32_t i = 0; i < n; ++i)
      {
        // TODO(wsl2): pull EWMA throughput, smoothed RTT, loss from path i.
        env.throughputMbps[i] = 0.0;
        env.rttMs[i] = 0.0;
        env.loss[i] = 0.0;
      }
  }

  // Deliver one segment over path `act.pathIdx`, advancing the simulation until
  // the transfer completes, then record realized stats into `env`.
  void DownloadSegment(const ActStruct& act, EnvStruct& env)
  {
    NS_ABORT_MSG_IF(act.pathIdx < 0 || act.pathIdx >= (int) m_cfg.paths.size(),
                    "pathIdx out of range");
    // TODO(wsl2):
    //   1) push `act.segmentBytes` onto subflow `act.pathIdx`;
    //   2) Simulator::Run() / a per-epoch stop event until it is delivered;
    //   3) record realized goodput/RTT/loss/duration below.
    env.lastPath = act.pathIdx;
    env.lastThroughputMbps = 0.0;
    env.lastRttMs = 0.0;
    env.lastLoss = 0.0;
    env.lastDurationS = 0.0;
    ++m_segmentsDone;
    env.done = (m_segmentsDone >= m_cfg.episodeSegments) ? 1 : 0;
  }

  uint32_t NumPaths() const { return static_cast<uint32_t>(m_cfg.paths.size()); }

private:
  ScenarioConfig m_cfg;
  NodeContainer m_client;
  NodeContainer m_server;
  std::vector<Ipv4InterfaceContainer> m_ifaces;
  uint32_t m_segmentsDone = 0;
};

// ---- ns3-ai decision loop -------------------------------------------------- //
//
// The episode is the per-segment exchange:
//   loop:
//     FillObservation(env);  send env -> Python;  block
//     recv act <- Python
//     DownloadSegment(act, env)
//     if env.done: break
//
// With the ns3-ai *message interface* this is a tight C++ loop using the
// blocking Get/Set buffer calls; with the ns3-ai *gym interface* it is driven by
// OpenGym callbacks. We use the message interface because the Python DataPlane
// (not C++) assembles the observation vector and reward — keeping C++ thin.

int
main(int argc, char* argv[])
{
  ScenarioConfig cfg;
  CommandLine cmd;
  cmd.AddValue("segments", "Segments per episode", cfg.episodeSegments);
  cmd.AddValue("segmentBytes", "Bytes per segment", cfg.segmentBytes);
  cmd.Parse(argc, argv);

  VideoMpquicScenario scenario(cfg);
  scenario.Build();

  // TODO(wsl2): construct the ns3-ai message interface with <EnvStruct, ActStruct>
  //   auto* iface = Ns3AiMsgInterface::Get();
  //   iface->SetIsMemoryCreator(false);
  //   iface->SetUseVector(false);
  //   iface->SetHandleFinish(true);
  //   Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>* msg = iface->GetInterface<EnvStruct, ActStruct>();

  EnvStruct env{};
  ActStruct act{};
  env.lastPath = -1;
  env.done = 0;

  for (uint32_t step = 0;; ++step)
    {
      scenario.FillObservation(env);

      // TODO(wsl2): publish `env` and block for the agent's action:
      //   msg->CppSendBegin(); msg->m_single_cpp2py_msg = env; msg->CppSendEnd();
      //   msg->CppRecvBegin(); act = msg->m_single_py2cpp_msg;  msg->CppRecvEnd();

      if (env.done)
        break;
      scenario.DownloadSegment(act, env);
      if (env.done)
        {
          // Publish terminal observation so Python sees done=1.
          scenario.FillObservation(env);
          env.done = 1;
          // TODO(wsl2): send terminal `env`.
          break;
        }
    }

  Simulator::Destroy();
  return 0;
}
