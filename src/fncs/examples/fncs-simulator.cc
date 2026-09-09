/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 INRIA
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cctype>
#include <cstdlib>
#include <limits>
#include "ns3/core-module.h"
#include "ns3/simulator.h"
#include "ns3/nstime.h"
#include "ns3/command-line.h"
#include "ns3/double.h"
#include "ns3/random-variable-stream.h"
#include "ns3/fncs-simulator-impl.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/bridge-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-global-routing-helper.h"

/**
 * \file
 * \ingroup simulator
 * Example program demonstrating use of various Schedule functions.
 */

using namespace ns3;

namespace
{

std::string
Trim (const std::string &s)
{
  size_t start = 0;
  while (start < s.size () && std::isspace (static_cast<unsigned char> (s[start])))
    {
      ++start;
    }
  size_t end = s.size ();
  while (end > start && std::isspace (static_cast<unsigned char> (s[end - 1])))
    {
      --end;
    }
  return s.substr (start, end - start);
}

bool
StartsWith (const std::string &s, const std::string &prefix)
{
  return s.rfind (prefix, 0) == 0;
}

std::string
StripQuotes (std::string s)
{
  s = Trim (s);
  if (s.size () >= 2)
    {
      if ((s.front () == '"' && s.back () == '"') || (s.front () == '\'' && s.back () == '\''))
        {
          return s.substr (1, s.size () - 2);
        }
    }
  return s;
}

std::pair<std::string, std::string>
SplitKeyValueOrDie (const std::string &line, uint32_t lineNo)
{
  auto pos = line.find (':');
  if (pos == std::string::npos)
    {
      NS_FATAL_ERROR ("YAML parse error at line " << lineNo << ": expected 'key: value' but got: "
                                             << line);
    }
  std::string key = Trim (line.substr (0, pos));
  std::string value = Trim (line.substr (pos + 1));
  if (key.empty ())
    {
      NS_FATAL_ERROR ("YAML parse error at line " << lineNo << ": empty key");
    }
  return {key, StripQuotes (value)};
}

std::vector<std::string>
SplitList (const std::string &value)
{
  std::string s = Trim (value);
  if (s.empty ())
    {
      return {};
    }

  // Support both YAML flow sequence: [a, b]  and simple CSV: a,b
  if (s.front () == '[' && s.back () == ']')
    {
      s = s.substr (1, s.size () - 2);
    }

  std::vector<std::string> out;
  std::string token;
  std::istringstream iss (s);
  while (std::getline (iss, token, ','))
    {
      token = StripQuotes (Trim (token));
      if (!token.empty ())
        {
          out.push_back (token);
        }
    }
  return out;
}

Time
ParseTimeOrDie (const std::string &s, const std::string &what)
{
  std::string t = Trim (s);
  if (t.empty ())
    {
      NS_FATAL_ERROR ("Missing time value for " << what);
    }

  // Accept forms like: 10s, 2ms, 50us, 100ns
  auto endsWith = [&] (const std::string &suffix) {
    return t.size () >= suffix.size () && t.compare (t.size () - suffix.size (), suffix.size (), suffix) == 0;
  };

  double scale = 0.0;
  std::string numberPart;
  if (endsWith ("ns"))
    {
      scale = 1e-9;
      numberPart = t.substr (0, t.size () - 2);
    }
  else if (endsWith ("us"))
    {
      scale = 1e-6;
      numberPart = t.substr (0, t.size () - 2);
    }
  else if (endsWith ("ms"))
    {
      scale = 1e-3;
      numberPart = t.substr (0, t.size () - 2);
    }
  else if (endsWith ("s"))
    {
      scale = 1.0;
      numberPart = t.substr (0, t.size () - 1);
    }
  else
    {
      NS_FATAL_ERROR ("Unsupported time unit for " << what << ": '" << t
                                                    << "' (use ns/us/ms/s)");
    }

  double v = 0.0;
  try
    {
      v = std::stod (Trim (numberPart));
    }
  catch (...)
    {
      NS_FATAL_ERROR ("Invalid numeric time for " << what << ": '" << t << "'");
    }
  return Seconds (v * scale);
}

struct NodeSpec
{
  std::string id;
  std::string type; // host | switch
};

struct LinkSpec
{
  std::string type; // p2p | csma
  std::vector<std::string> endpoints;
  std::string dataRate = "5Mbps";
  std::string delay = "2ms";
};

struct AppSpec
{
  std::string type; // fncs | udpecho-server | udpecho-client
  std::unordered_map<std::string, std::string> kv;
};

struct TopologySpec
{
  Time stop = Seconds (10.0);
  std::vector<NodeSpec> nodes;
  std::vector<LinkSpec> links;
  std::vector<AppSpec> apps;
};

TopologySpec
LoadTopologyYamlOrDie (const std::string &path)
{
  std::ifstream in (path);
  if (!in.is_open ())
    {
      NS_FATAL_ERROR ("Unable to open topology YAML: " << path);
    }

  enum class Section
  {
    None,
    Sim,
    Nodes,
    Links,
    Apps
  };

  TopologySpec spec;
  Section section = Section::None;

  std::unordered_map<std::string, std::string> currentItem;
  auto flushItem = [&] () {
    if (currentItem.empty ())
      {
        return;
      }
    if (section == Section::Nodes)
      {
        NodeSpec n;
        n.id = currentItem["id"];
        n.type = currentItem.count ("type") ? currentItem["type"] : currentItem["kind"];
        n.id = Trim (n.id);
        n.type = Trim (n.type);
        if (n.id.empty ())
          {
            NS_FATAL_ERROR ("Node is missing 'id'");
          }
        if (n.type.empty ())
          {
            NS_FATAL_ERROR ("Node '" << n.id << "' is missing 'type' (host|switch)");
          }
        spec.nodes.push_back (n);
      }
    else if (section == Section::Links)
      {
        LinkSpec l;
        l.type = currentItem["type"];
        if (currentItem.count ("dataRate"))
          {
            l.dataRate = currentItem["dataRate"];
          }
        if (currentItem.count ("delay"))
          {
            l.delay = currentItem["delay"];
          }
        if (currentItem.count ("endpoints"))
          {
            l.endpoints = SplitList (currentItem["endpoints"]);
          }
        if (Trim (l.type).empty ())
          {
            NS_FATAL_ERROR ("Link is missing 'type' (p2p|csma)");
          }
        if (l.endpoints.size () < 2)
          {
            NS_FATAL_ERROR ("Link 'endpoints' must have at least 2 node ids");
          }
        spec.links.push_back (l);
      }
    else if (section == Section::Apps)
      {
        AppSpec a;
        a.type = currentItem["type"];
        if (Trim (a.type).empty ())
          {
            NS_FATAL_ERROR ("App is missing 'type'");
          }
        a.kv = std::move (currentItem);
        spec.apps.push_back (a);
      }
    currentItem.clear ();
  };

  std::string raw;
  uint32_t lineNo = 0;
  while (std::getline (in, raw))
    {
      ++lineNo;
      // Strip comments (# ...)
      auto hash = raw.find ('#');
      if (hash != std::string::npos)
        {
          raw = raw.substr (0, hash);
        }
      std::string line = Trim (raw);
      if (line.empty ())
        {
          continue;
        }

      if (line == "sim:")
        {
          flushItem ();
          section = Section::Sim;
          continue;
        }
      if (line == "nodes:")
        {
          flushItem ();
          section = Section::Nodes;
          continue;
        }
      if (line == "links:")
        {
          flushItem ();
          section = Section::Links;
          continue;
        }
      if (line == "apps:")
        {
          flushItem ();
          section = Section::Apps;
          continue;
        }

      if (section == Section::Sim)
        {
          auto kv = SplitKeyValueOrDie (line, lineNo);
          if (kv.first == "stop")
            {
              spec.stop = ParseTimeOrDie (kv.second, "sim.stop");
            }
          else
            {
              NS_FATAL_ERROR ("Unknown sim key at line " << lineNo << ": " << kv.first);
            }
          continue;
        }

      if (section == Section::Nodes || section == Section::Links || section == Section::Apps)
        {
          if (StartsWith (line, "-"))
            {
              flushItem ();
              std::string rest = Trim (line.substr (1));
              if (!rest.empty ())
                {
                  auto kv = SplitKeyValueOrDie (rest, lineNo);
                  currentItem[kv.first] = kv.second;
                }
              continue;
            }

          auto kv = SplitKeyValueOrDie (line, lineNo);
          currentItem[kv.first] = kv.second;
          continue;
        }

      NS_FATAL_ERROR ("YAML parse error at line " << lineNo << ": unexpected content outside sections: "
                                                   << line);
    }

  flushItem ();
  return spec;
}

} // namespace

class MyModel
{
public:
  void Start (void);
private:
  void HandleEvent (double eventValue);
};

void
MyModel::Start (void)
{
  Simulator::Schedule (Seconds (10.0),
                       &MyModel::HandleEvent,
                       this, Simulator::Now ().GetSeconds ());
}
void
MyModel::HandleEvent (double value)
{
  std::cout << "Member method received event at "
            << Simulator::Now ().GetSeconds ()
            << "s started at " << value << "s" << std::endl;
}

static void
ExampleFunction (MyModel *model)
{
  std::cout << "ExampleFunction received event at "
            << Simulator::Now ().GetSeconds () << "s" << std::endl;
  model->Start ();
}

static void
RandomFunction (void)
{
  std::cout << "RandomFunction received event at "
            << Simulator::Now ().GetSeconds () << "s" << std::endl;
}

static void
CancelledEvent (void)
{
  std::cout << "I should never be called... " << std::endl;
}

int main (int argc, char *argv[])
{
  std::string topoYaml;

  CommandLine cmd;
  cmd.AddValue ("topo", "Topology YAML file (enables YAML-driven topology)", topoYaml);
  cmd.Parse (argc, argv);


  // Fncs simulation setup
  Ptr<FncsSimulatorImpl> sim = CreateObject<FncsSimulatorImpl> ();
  Simulator::SetImplementation(sim);
  LogComponentEnable ("FncsApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("FncsSimulatorImpl", LOG_LEVEL_INFO);

    Time::SetResolution(Time::NS);
    LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    // Default: preserve original 2-node p2p + UDP echo setup when no YAML is provided.
    if (topoYaml.empty ())
      {
        std::cout << "No topology YAML provided, running default 2-node p2p + UDP echo example." << std::endl;
        NodeContainer nodes;
        nodes.Create (2);

        PointToPointHelper pointToPoint;
        pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
        pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

        NetDeviceContainer devices;
        devices = pointToPoint.Install (nodes);

        InternetStackHelper stack;
        stack.Install (nodes);

        Ipv4AddressHelper address;
        address.SetBase ("10.1.1.0", "255.255.255.0");

        Ipv4InterfaceContainer interfaces = address.Assign (devices);

        UdpEchoServerHelper echoServer (9);

        FncsApplicationHelper fncsHelper ("ns3::FncsServer", 1);

        ApplicationContainer fncsApps = fncsHelper.Install (nodes.Get (0), "node1");
        fncsApps.Add (fncsHelper.Install (nodes.Get (1), "node2"));
        fncsApps.Start (Seconds (0));
        fncsApps.Stop (Seconds (INT_MAX));

        ApplicationContainer serverApps = echoServer.Install (nodes.Get (1));
        serverApps.Start (Seconds (0));
        serverApps.Stop (Seconds (10));

        UdpEchoClientHelper echoClient (interfaces.GetAddress (1), 9);
        echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
        echoClient.SetAttribute ("Interval", TimeValue (Seconds (1)));
        echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

        ApplicationContainer clientApps = echoClient.Install (nodes.Get (0));
        clientApps.Start (Seconds (0));
        clientApps.Stop (Seconds (10));

        Simulator::Run ();
        Simulator::Destroy ();
        return 0;
      }

    // YAML-driven topology
    TopologySpec spec = LoadTopologyYamlOrDie (topoYaml);

    std::unordered_map<std::string, Ptr<Node>> nodesById;
    std::unordered_set<std::string> switches;
    NodeContainer hostNodes;
    for (const auto &n : spec.nodes)
      {
        Ptr<Node> node = CreateObject<Node> ();
        nodesById[n.id] = node;
        if (n.type == "switch")
          {
            switches.insert (n.id);
          }
        else if (n.type == "host")
          {
            hostNodes.Add (node);
          }
        else
          {
            NS_FATAL_ERROR ("Unknown node type for '" << n.id << "': '" << n.type << "' (use host|switch)");
          }
      }

    InternetStackHelper stack;
    stack.Install (hostNodes);

    // For bridge switches, collect their CSMA ports and the host-side devices to address as one LAN.
    std::unordered_map<std::string, NetDeviceContainer> switchPorts;
    std::unordered_map<std::string, NetDeviceContainer> switchLanHostDevices;
    std::vector<NetDeviceContainer> standaloneCsmaLans; // LANs without switches

    // P2P links can be addressed immediately (each link is its own subnet).
    uint32_t subnetIndex = 1;
    auto nextSubnetBase = [&] () {
      std::ostringstream os;
      os << "10.1." << subnetIndex++ << ".0";
      return os.str ();
    };

    for (const auto &l : spec.links)
      {
        if (l.type == "p2p")
          {
            if (l.endpoints.size () != 2)
              {
                NS_FATAL_ERROR ("p2p link requires exactly 2 endpoints");
              }
            const std::string &a = l.endpoints[0];
            const std::string &b = l.endpoints[1];
            if (!nodesById.count (a) || !nodesById.count (b))
              {
                NS_FATAL_ERROR ("p2p link references unknown endpoint(s): " << a << ", " << b);
              }

            PointToPointHelper p2p;
            p2p.SetDeviceAttribute ("DataRate", StringValue (l.dataRate));
            p2p.SetDeviceAttribute ("Mtu", UintegerValue (64000));
            p2p.SetChannelAttribute ("Delay", StringValue (l.delay));
            p2p.SetQueue ("ns3::DropTailQueue", "MaxSize", StringValue ("20000p"));

            NetDeviceContainer devices = p2p.Install (nodesById[a], nodesById[b]);
            Ipv4AddressHelper address;
            address.SetBase (nextSubnetBase ().c_str (), "255.255.255.0");
            address.Assign (devices);
          }
        else if (l.type == "csma")
          {
            // If one endpoint is a switch and there are exactly two endpoints, treat it as a port on a bridge.
            std::string switchId;
            for (const auto &e : l.endpoints)
              {
                if (switches.count (e))
                  {
                    switchId = e;
                    break;
                  }
              }

            CsmaHelper csma;
            csma.SetChannelAttribute ("DataRate", StringValue (l.dataRate));
            csma.SetChannelAttribute ("Delay", TimeValue (ParseTimeOrDie (l.delay, "links.delay")));
            csma.SetDeviceAttribute ("Mtu", UintegerValue (64000));
            csma.SetQueue ("ns3::DropTailQueue", "MaxSize", StringValue ("20000p"));

            if (!switchId.empty () && l.endpoints.size () == 2)
              {
                const std::string &a = l.endpoints[0];
                const std::string &b = l.endpoints[1];
                std::string hostId = (a == switchId) ? b : a;
                if (!nodesById.count (hostId) || !nodesById.count (switchId))
                  {
                    NS_FATAL_ERROR ("csma link references unknown endpoint(s)");
                  }
                if (!switches.count (switchId))
                  {
                    NS_FATAL_ERROR ("Internal error: expected '" << switchId << "' to be a switch");
                  }

                NodeContainer pair;
                pair.Add (nodesById[hostId]);
                pair.Add (nodesById[switchId]);
                NetDeviceContainer devices = csma.Install (pair);

                // devices[0] corresponds to hostId, devices[1] to switchId (pair order).
                switchLanHostDevices[switchId].Add (devices.Get (0));
                switchPorts[switchId].Add (devices.Get (1));
              }
            else
              {
                // Standalone CSMA LAN (no bridge switch). All endpoints must be hosts.
                NodeContainer lan;
                for (const auto &e : l.endpoints)
                  {
                    if (!nodesById.count (e))
                      {
                        NS_FATAL_ERROR ("csma link references unknown endpoint: " << e);
                      }
                    if (switches.count (e))
                      {
                        NS_FATAL_ERROR ("csma LAN without 1:1 switch port mapping does not support endpoint of type switch: "
                                        << e);
                      }
                    lan.Add (nodesById[e]);
                  }
                NetDeviceContainer devices = csma.Install (lan);
                standaloneCsmaLans.push_back (devices);
              }
          }
        else
          {
            NS_FATAL_ERROR ("Unknown link type: '" << l.type << "' (use p2p|csma)");
          }
      }

    // Install bridges on switch nodes.
    for (const auto &kv : switchPorts)
      {
        const std::string &switchId = kv.first;
        Ptr<Node> sw = nodesById[switchId];
        BridgeHelper bridge;
        bridge.Install (sw, kv.second);
      }

    // Address assignment: each bridge switch forms one LAN subnet; each standalone CSMA LAN is one subnet.
    for (auto &kv : switchLanHostDevices)
      {
        Ipv4AddressHelper address;
        address.SetBase (nextSubnetBase ().c_str (), "255.255.255.0");
        address.Assign (kv.second);
      }
    for (auto &lanDevices : standaloneCsmaLans)
      {
        Ipv4AddressHelper address;
        address.SetBase (nextSubnetBase ().c_str (), "255.255.255.0");
        address.Assign (lanDevices);
      }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

    // Install apps
    ApplicationContainer allApps;
    for (const auto &a : spec.apps)
      {
        if (a.type == "fncs")
          {
            auto itNode = a.kv.find ("node");
            auto itName = a.kv.find ("name");
            if (itNode == a.kv.end () || itName == a.kv.end ())
              {
                NS_FATAL_ERROR ("fncs app requires keys: node, name");
              }
            const std::string &nodeId = itNode->second;
            const std::string &name = itName->second;
            if (!nodesById.count (nodeId))
              {
                NS_FATAL_ERROR ("fncs app references unknown node: " << nodeId);
              }
            FncsApplicationHelper fncsHelper ("ns3::FncsServer", 1);
            allApps.Add (fncsHelper.Install (nodesById[nodeId], name));
          }
        else if (a.type == "udpecho-server")
          {
            auto itNode = a.kv.find ("node");
            auto itPort = a.kv.find ("port");
            if (itNode == a.kv.end () || itPort == a.kv.end ())
              {
                NS_FATAL_ERROR ("udpecho-server app requires keys: node, port");
              }
            const std::string &nodeId = itNode->second;
            uint16_t port = static_cast<uint16_t> (std::stoul (itPort->second));
            if (!nodesById.count (nodeId))
              {
                NS_FATAL_ERROR ("udpecho-server app references unknown node: " << nodeId);
              }
            UdpEchoServerHelper server (port);
            allApps.Add (server.Install (nodesById[nodeId]));
          }
        else if (a.type == "udpecho-client")
          {
            auto itNode = a.kv.find ("node");
            auto itRemote = a.kv.find ("remote");
            auto itPort = a.kv.find ("port");
            if (itNode == a.kv.end () || itRemote == a.kv.end () || itPort == a.kv.end ())
              {
                NS_FATAL_ERROR ("udpecho-client app requires keys: node, remote, port");
              }
            const std::string &nodeId = itNode->second;
            const std::string &remoteId = itRemote->second;
            uint16_t port = static_cast<uint16_t> (std::stoul (itPort->second));
            if (!nodesById.count (nodeId) || !nodesById.count (remoteId))
              {
                NS_FATAL_ERROR ("udpecho-client app references unknown node(s): " << nodeId << ", " << remoteId);
              }

            Ptr<Ipv4> remoteIpv4 = nodesById[remoteId]->GetObject<Ipv4> ();
            Ipv4InterfaceAddress ifAddr = remoteIpv4->GetAddress (1, 0);
            Ipv4Address remoteAddr = ifAddr.GetLocal ();

            UdpEchoClientHelper client (remoteAddr, port);
            if (a.kv.count ("maxPackets"))
              {
                client.SetAttribute ("MaxPackets", UintegerValue (std::stoul (a.kv.at ("maxPackets"))));
              }
            if (a.kv.count ("interval"))
              {
                client.SetAttribute ("Interval", TimeValue (ParseTimeOrDie (a.kv.at ("interval"), "udpecho.interval")));
              }
            if (a.kv.count ("packetSize"))
              {
                client.SetAttribute ("PacketSize", UintegerValue (std::stoul (a.kv.at ("packetSize"))));
              }
            allApps.Add (client.Install (nodesById[nodeId]));
          }
        else
          {
            NS_FATAL_ERROR ("Unknown app type: '" << a.type
                                                  << "' (use fncs|udpecho-server|udpecho-client)");
          }
      }

    allApps.Start (Seconds (0));
    allApps.Stop (Seconds (INT_MAX));

    Simulator::Stop (spec.stop);

    Simulator::Run();
    Simulator::Destroy();
    return 0;
  
  // //Define jitter parameters to simulate lack of total synchronicity in all objects
  // Config::SetDefault ("ns3::FncsApplication::JitterMinNs", DoubleValue (10));
  // Config::SetDefault ("ns3::FncsApplication::JitterMaxNs", DoubleValue (100));

  // MyModel model;
  // Ptr<UniformRandomVariable> v = CreateObject<UniformRandomVariable> ();
  // v->SetAttribute ("Min", DoubleValue (10));
  // v->SetAttribute ("Max", DoubleValue (20));

  // Simulator::Schedule (Seconds (10.0), &ExampleFunction, &model);

  // Simulator::Schedule (Seconds (v->GetValue ()), &RandomFunction);

  // EventId id = Simulator::Schedule (Seconds (30.0), &CancelledEvent);
  // Simulator::Cancel (id);

  // // schedule when to end the simulation
  // Simulator::Stop (Seconds (100.0));

  // //开始设置拓扑


  // Simulator::Run ();

  // Simulator::Destroy ();
}
