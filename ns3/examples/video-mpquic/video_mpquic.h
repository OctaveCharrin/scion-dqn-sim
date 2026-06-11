/*
 * video_mpquic.h — shared-memory message structs for the RL-controlled
 * adaptive-video-over-(abstracted)-multipath scenario.
 *
 * These two structs are the wire format of the ns3-ai shared-memory bridge.
 * They are included by BOTH the C++ scenario (video_mpquic.cc) and the pybind11
 * binding (video_mpquic_py.cc) so the layout is identical on both sides, and
 * they mirror the Python dataclasses PathStats / DownloadResult in
 * src/ns3env/dataplane.py.
 *
 * Decision epoch = ONE video segment download.
 */

#ifndef VIDEO_MPQUIC_H
#define VIDEO_MPQUIC_H

#include <cstdint>

// Upper bound on candidate paths (subflows). Fixed-size arrays keep the shared
// memory layout POD/trivially-copyable, which the struct-based msg interface
// requires. Must match kMaxPaths usage in Python (we only read numPaths slots).
static constexpr uint32_t kMaxPaths = 8;

// C++ -> Python: observable per-path state at a decision point, plus the
// realized result of the *previous* segment (lastPath == -1 before the first).
struct EnvStruct
{
    uint32_t numPaths;
    double clockS; // current sim time (s)
    uint8_t done;  // 1 once the episode horizon is reached

    // Per-path observation (length == numPaths). Mirrors PathStats.
    double throughputMbps[kMaxPaths]; // EWMA realized goodput estimate per path
    double rttMs[kMaxPaths];          // smoothed RTT per path
    double loss[kMaxPaths];           // loss estimate per path (0..1)

    // Result of the segment delivered since the previous decision. Mirrors
    // DownloadResult. lastPath == -1 on the initial observation.
    int32_t lastPath;
    double lastThroughputMbps; // realized goodput for that segment
    double lastRttMs;
    double lastLoss;
    double lastDurationS;     // sim time the download took
    uint32_t lastBytes;       // bytes delivered
};

// Action commands. One NS-3 process serves many episodes (the simulation is
// continuing), so Python drives episode boundaries in-band rather than by
// relaunching the process (ns3-ai allows only one shared-memory creator per
// Python process).
enum ActCommand : int32_t
{
    ACT_STEP = 0,      // download a segment on pathIdx
    ACT_RESET = 1,     // start a new episode (reset counters; keep sim running)
    ACT_TERMINATE = 2, // end the NS-3 process
};

// Python -> C++: the agent's action for the next decision.
struct ActStruct
{
    int32_t command;       // one of ActCommand
    int32_t pathIdx;       // which subflow to fetch the next segment on (ACT_STEP)
    uint32_t segmentBytes; // segment size to deliver this epoch
    // Phase 2+: extend with sendRateMbps / bitrateLevel here.
};

#endif // VIDEO_MPQUIC_H
