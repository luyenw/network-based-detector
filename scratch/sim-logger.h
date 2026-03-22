#pragma once
/* =============================================================================
 * sim-logger.h  —  Structured file + console logger
 *
 * • Works in ALL ns-3 build profiles (optimized / debug / default)
 *   because it uses only std:: — no NS_LOG dependency.
 * • Writes every line to stdout (visible in docker logs) AND to a file
 *   under /output/ (persists after container exits).
 * • Optional simulation-time annotation via simTimeGetter callback.
 *
 * Quick start:
 *   // In main(), early:
 *   SimLogger::Get().Open("/output/simulation.log");
 *   SimLogger::Get().simTimeGetter = [](){
 *       return ns3::Simulator::Now().GetSeconds();
 *   };
 *   SimLogger::Get().minLevel = SimLogLevel::DEBUG; // for full detail
 *
 *   // Anywhere:
 *   SIM_LOG(INFO,  "Setup",   "gridSize=" << g << " numUe=" << n);
 *   SIM_LOG(DEBUG, "Callback","IMSI=" << i << " rsrp=" << r << "dBm");
 *   SIM_LOG(WARN,  "CcPCV",   "cell " << id << " skipped (no position)");
 *   SIM_LOG(ERR,   "FileIO",  "cannot open " << path);
 *
 *   // In main(), at the end:
 *   SimLogger::Get().Close();
 * =============================================================================
 */

#include <chrono>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Log levels
// ---------------------------------------------------------------------------
enum class SimLogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERR = 3 };

// ---------------------------------------------------------------------------
// SimLogger singleton
// ---------------------------------------------------------------------------
class SimLogger
{
public:
    // Minimum level to emit (change to DEBUG for full trace)
    SimLogLevel minLevel = SimLogLevel::INFO;

    // Optional: set to a lambda returning the current simulation time [s].
    // When set, every log line includes "[sim=X.XXXs]".
    std::function<double()> simTimeGetter;

    // ---- Singleton accessor ------------------------------------------------
    static SimLogger& Get()
    {
        static SimLogger instance;
        return instance;
    }

    // ---- Open log file -----------------------------------------------------
    void Open(const std::string& path)
    {
        m_ofs.open(path, std::ios::out | std::ios::trunc);
        m_open = m_ofs.is_open();
        if (!m_open)
        {
            std::cerr << "[SimLogger] WARNING: cannot open log file: "
                      << path << "\n";
            return;
        }
        m_ofs << "# ============================================================\n"
              << "# Fake-BS Simulation Log\n"
              << "# Format: [wall HH:MM:SS.mmm] [LEVEL] [sim=Xs] [Component    ] Message\n"
              << "# Started: " << WallDateTimeStr() << "\n"
              << "# ============================================================\n\n";
        m_ofs.flush();
    }

    // ---- Emit a log line ---------------------------------------------------
    void Emit(SimLogLevel level,
              const std::string& comp,
              const std::string& msg)
    {
        if (level < minLevel) return;
        std::string line = FormatLine(level, comp, msg);
        // stdout — always visible in `docker compose logs`
        std::cout << line << "\n";
        std::cout.flush();
        // file — persists in /output/ after container exits
        if (m_open)
        {
            m_ofs << line << "\n";
            m_ofs.flush();
        }
    }

    // ---- Close log file ----------------------------------------------------
    void Close()
    {
        if (!m_open) return;
        m_ofs << "\n# ============================================================\n"
              << "# Ended: " << WallDateTimeStr() << "\n"
              << "# ============================================================\n";
        m_ofs.close();
        m_open = false;
    }

private:
    std::ofstream m_ofs;
    bool          m_open = false;

    SimLogger() = default;

    // Wall-clock HH:MM:SS.mmm
    std::string WallTimeStr() const
    {
        auto tp = std::chrono::system_clock::now();
        auto tt = std::chrono::system_clock::to_time_t(tp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      tp.time_since_epoch()) % 1000;
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        std::ostringstream ss;
        ss << std::put_time(&tm, "%H:%M:%S")
           << "." << std::setw(3) << std::setfill('0') << ms.count();
        return ss.str();
    }

    // Wall-clock YYYY-MM-DD HH:MM:SS for header
    std::string WallDateTimeStr() const
    {
        auto tp = std::chrono::system_clock::now();
        auto tt = std::chrono::system_clock::to_time_t(tp);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        std::ostringstream ss;
        ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    const char* LevelStr(SimLogLevel level) const
    {
        switch (level)
        {
            case SimLogLevel::DEBUG: return "DEBUG";
            case SimLogLevel::INFO:  return "INFO ";
            case SimLogLevel::WARN:  return "WARN ";
            case SimLogLevel::ERR:   return "ERROR";
            default:                 return "?????";
        }
    }

    std::string FormatLine(SimLogLevel    level,
                            const std::string& comp,
                            const std::string& msg) const
    {
        std::ostringstream ss;
        ss << "[" << WallTimeStr() << "]"
           << " [" << LevelStr(level) << "]";

        if (simTimeGetter)
            ss << " [sim=" << std::fixed << std::setprecision(3)
               << simTimeGetter() << "s]";

        // Left-align component name in a fixed-width field for readability
        ss << " [" << std::setw(16) << std::left << comp << "] " << msg;
        return ss.str();
    }
};

// ---------------------------------------------------------------------------
// SIM_LOG(LEVEL, "Component", stream_expression)
//
// The if-check uses the enum directly so the compiler can eliminate the
// entire branch when minLevel > level (no ostringstream allocation).
//
// Examples:
//   SIM_LOG(INFO,  "FakeBsSim",  "gridSize=" << g << " isd=" << isd);
//   SIM_LOG(DEBUG, "RsrpCb",     "IMSI=" << i << " t=" << t);
//   SIM_LOG(WARN,  "CcpcvDetect","skipping cell " << c << ": no position");
//   SIM_LOG(ERR,   "FileIO",     "cannot open: " << path);
// ---------------------------------------------------------------------------
#define SIM_LOG(level, comp, msg)                                   \
    do {                                                             \
        if (SimLogLevel::level >= SimLogger::Get().minLevel) {      \
            std::ostringstream _sl_ss_;                              \
            _sl_ss_ << msg;                                          \
            SimLogger::Get().Emit(SimLogLevel::level,                \
                                  (comp), _sl_ss_.str());            \
        }                                                            \
    } while (0)
