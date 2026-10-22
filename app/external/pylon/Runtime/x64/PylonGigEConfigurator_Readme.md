# Overview of the pylon GigE Configurator

This topic gives you an overview of the pylon GigE Configurator.

The pylon GigE Configurator is part of the [Basler pylon Camera Software Suite](https://www.baslerweb.com/en/products/software/). The tool optimizes the parameters and settings of your network adapters and cameras for your operating system (Windows or Linux).

You can use the GigE Configurator either via pylon Viewer (under **Tools**) for standard use cases, or via its command line interface (CLI) for advanced use cases.

## Opening the Command Line Version of the GigE Configurator

To open the command line version of the pylon GigE Configurator:

1. Make sure pylon version 7.1 or higher is installed.
2. Open a command line.
3. On the command line, navigate to your pylon installation directory, e.g., **%programfiles%\Basler\pylon x\Runtime\x64** (Windows) or **/opt/pylon x/bin** (Linux).
4. Run **PylonGigEConfigurator** from the command line.

!!! info
    * For a local network adapter configuration, the pylon GigE Configurator needs to be started with admin or sudo rights.
    * To avoid network conflicts, make sure only cameras are connected to your network adapter or switch.

## Supported Operating Systems

The pylon GigE Configurator supports the following operating systems:

* Windows 64 bit
* Linux x86_64
* Linux AArch64

!!! info
    On Linux, the **ethtool** and **network-manager** packages must be installed before running the pylon GigE Configurator. If necessary, install them using following commands:
    ```
    sudo apt install network-manager
    sudo apt install ethtool
    ```

## Supported GigE Adapters

The pylon GigE Configurator supports the following GigE adapters:

* All 1 GigE network adapters
* All Basler 5/10 GigE network adapters

For information about Basler's network adapters, see the [Basler website](https://www.baslerweb.com/en/products/acquisition-cards/gige-interface-cards/#accessory_overview__pc_karten).

## Supported Cameras

The pylon GigE Configurator supports all GigE cameras manufactured by Basler AG.

## Commands

The tool accepts the following commands and options:

| Command    | Explanation                                                                                                                                         |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `list`     | Displays a list of active network adapters and their current configuration.                                                                         |
| `auto-ip`  | Configures the IP addresses and subnet masks of network adapters and attached cameras.                                                              |
| `auto-opt` | Optimizes network adapters and system settings (e.g., Jumbo Frames, Interrupt Moderation Rate, Receive Descriptors) for best streaming performance. |
| `auto-all` | Runs both auto-opt and auto-ip.                                                                                                                     |

| Option                         | Explanation                                                                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `-h` or `--help`               | Show help.                                                                                                                              |
| `-l` or `--log <filename>`     | Log to the file specified. Defaults to **<TEMP\>/<appname\>_<timestamp\>.log**.                                                         |
| `-n` or `--dry-run`            | [Dry-run mode](#dry-run-mode-of-the-pylon-gige-configurator). Don't actually set the parameters, only validate them.                    |
| `-a` or `--adaptername <name>` | Enforce assigning an IP address and/or optimizing the settings of the specified adapter.                                                |
| `--class <A, B, C>`            | When setting IP addresses, use private addresses from the specified network class. Valid network classes are A, B, or C. Defaults to C. |

### Examples

```bash
PylonGigEConfigurator list

PylonGigEConfigurator auto-all -h

PylonGigEConfigurator auto-ip -a "Ethernet 2" --class C
```

# Optimizing the System Using the pylon GigE Configurator

The auto-all command optimizes your GigE network and camera setup in a single step.

| Command Syntax                                              |
| ----------------------------------------------------------- |
| `PylonGigEConfigurator auto-all [-a <name> ...] [--class <A | B | C>] [-n] [-h] [-l <filename>]` |

## How It Works

Executing the `auto-all` command does the following:

1. Runs the `auto-ip` command to [configure the IP addresses](#configuring-the-ip-address-using-the-pylon-gige-configurator) and subnet masks of network adapters and attached cameras.
2. Runs the `auto-opt` command to [configure your network adapters](#configuring-network-settings-using-the-pylon-gige-configurator) and your system settings for best streaming performance.

For more information, see the documentation for both commands.

# Configuring the IP Address Using the pylon GigE Configurator

The auto-ip command configures the IP addresses and subnet masks of network adapters and attached cameras.

| Command Syntax                                             |
| ---------------------------------------------------------- |
| `PylonGigEConfigurator auto-ip [-a <name> ...] [--class <A | B | C>] [-n] [-h] [-l <filename>]` |

The `auto-ip` command scans all local GigE network adapters and checks for connected cameras. If no camera has been detected, the given adapter is skipped and won't be configured.

Then, the tool configures the IP addresses of all GigE network adapters with connected cameras in ascending order.

You can force the IP configuration of a network adapter by using the `-a` option, followed by the adapter's name. In this case, the adapter will always be configured, regardless of whether a camera is connected or not.

Also, you can configure a network adapter within other class address ranges, e.g., class **A** or class **B**. To do so, use the the `--class` option followed by the class type, e.g., `--class B`.

If you omit the `--class` option, the tool automatically uses the address range from network class **C**, e.g., 192.168.xxx.xxx.

## Examples

```
PylonGigEConfigurator auto-ip
PylonGigEConfigurator auto-ip -a "Ethernet 2" -a "Ethernet 3"
PylonGigEConfigurator auto-ip -a "Ethernet 2" --class C
```

# Configuring Network Settings Using the pylon GigE Configurator

The auto-opt command optimizes your network adapters and your system settings for best streaming performance.

| Command Syntax                                                             |
| -------------------------------------------------------------------------- |
| `PylonGigEConfigurator auto-opt [-a <name> ...] [-n] [-h] [-l <filename>]` |

The `auto-opt` command scans all local GigE network adapters and checks for connected cameras. If no camera has been detected, the given adapter is skipped and won't be configured.

Then, the tool optimizes all GigE network adapters with connected cameras in ascending order.

You can force the optimization of a network adapter by using the `-a` option, followed by the adapter's name. In this case, the adapter will always be configured, regardless of whether a camera is connected or not.

## Optimized Parameters

### Windows

On Windows, the `auto-opt` command optimizes the following parameters:

| Parameters                | Description                                                                                                                                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Jumbo Packet Size         | Size of the Jumbo Packets, i.e., Ethernet packets that exceed the standard packet size of 1518 bytes.<br/>Will be set to the **maximum** possible value.                                               |
| Receive Buffers           | Buffer size of system memory that can be used by the adapter for received packets.<br/>Will be set to the **maximum** possible value, or to **2048** if the driver doesn't provide a registry setting. |
| Interrupt Moderation      | Will be set to `ON`, if available.                                                                                                                                                                     |
| Interrupt Moderation Rate | Number of interrupts per second.<br/>Will be set to **Extreme**, or to the current value if the driver doesn't provide a registry setting.                                                             |

### Linux

On Linux, the `auto-opt` command optimizes the following parameters:

| Parameters                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Maximum Transmission Unit | Largest size of an Ethernet packet that can be sent over a TCP/IP networking connection.<br/>Will be set to the **maximum** possible value.                                                                                                                                                                                                                                                                                                                            |
| rtprio                    | Modifies whether a process has realtime or idle priority.<br/>Will be set to **99** in **/etc/security/limits.conf**.                                                                                                                                                                                                                                                                                                                                                  |
| rp_filter                 | Checks the routing table against the source address of incoming packets. This ensures that packets are coming from the interface as defined in the routing table.<br/>In **/etc/sysctl.conf**, the `net.ipv4.conf.<network name>.rp_filter` and `net.ipv4.conf.all.rp_filter` values will be set to **0**.                                                                                                                                                             |
| rmem_max                  | Size of the buffer that receives UDP packets.<br/>In **/etc/sysctl.conf**, the `net.core.rmem_max` value will be set to **33554432**.                                                                                                                                                                                                                                                                                                                                  |
| ringbuffer                | The ring buffer is a circular buffer that stores incoming packets to prevent buffer overflow. For this parameter to be set, you must install **ethtool** first (see [Supported Operating Systems](#supported-operating-systems).<br/>Will be set to **4096** for every GigE adapter. The pylon GigE Configurator also adds or modifies a network startup script in **/etc/NetworkManager/dispatcher.d/pre-up.d/basler-network-config** to make the settings permanent. |
| Interrupt Moderation Rate | Number of interrupts per second. For this parameter to be set, you must install **ethtool** first (see [Supported Operating Systems](#supported-operating-systems).<br/>Will be set to **84** for every GigE adapter. pylon GigE Configurator also adds or modifies a network startup script in **/etc/NetworkManager/dispatcher.d/pre-up.d/basler-network-config** to make the settings permanent.                                                                    |

!!! info
    You may have to reboot your computer after optimization (Linux only).

## Examples

```
PylonGigEConfigurator auto-opt

PylonGigEConfigurator auto-opt -a "Ethernet 2"
```

# Dry-run Mode of the pylon GigE Configurator

The dry-run mode of the pylon GigE Configurator allows you validate the changes to your system before applying them.

In dry-run mode, the pylon GigE Configurator shows you all configuration and optimization steps that will be performed when you execute the tool.

You can enable dry-run mode by using the `-n` or `--dry-run` option with any command.

## Examples

```
PylonGigEConfigurator auto-all -n

PylonGigEConfigurator auto-ip -n

PylonGigEConfigurator auto-opt -n
```

# Logging Changes Made by the pylon GigE Configurator

The pylon GigE Configurator allows you to create log files for diagnostics.

| Command Syntax                                                             |
| -------------------------------------------------------------------------- |
| `PylonGigEConfigurator auto-opt [-a <name> ...] [-n] [-h] [-l <filename>`] |

You can log all changes made by pylon GigE Configurator with the `-l` or `--log` option. If you want to log the changes to a user-defined file and directory, append the file name, e.g., `-l log.txt`.

By default, pylon GigE Configurator logs all changes into a log file in the **temp** folder, e.g.,:

- Windows : **%TEMP%\PylonGigEConfigurator_<timestamp\>.log**
- Linux : **~/.local/temp/PylonGigEConfigurator_<timestamp\>.log**

## Example

```
PylonGigEConfigurator auto-all -l log.txt
```
