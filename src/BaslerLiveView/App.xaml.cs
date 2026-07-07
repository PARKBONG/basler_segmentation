using System;
using System.Windows;

namespace BaslerLiveView;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        // Optional: launch with `--emulate` (or set PYLON_CAMEMU) to expose
        // Basler's software camera emulator when no physical camera is present.
        // Must be set before any pylon API call (i.e. before the first enumerate).
        foreach (var arg in e.Args)
        {
            if (arg.Equals("--emulate", StringComparison.OrdinalIgnoreCase))
            {
                if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("PYLON_CAMEMU")))
                    Environment.SetEnvironmentVariable("PYLON_CAMEMU", "1");
            }
        }

        base.OnStartup(e);
    }
}
