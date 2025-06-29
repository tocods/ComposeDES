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
#include "ns3/point-to-point-module.h"
#include "ns3/ipv4.h"

/**
 * \file
 * \ingroup simulator
 * Example program demonstrating use of various Schedule functions.
 */

using namespace ns3;

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
  CommandLine cmd;
  cmd.Parse (argc, argv);


  // Fncs simulation setup
  Ptr<FncsSimulatorImpl> sim = CreateObject<FncsSimulatorImpl> ();
  Simulator::SetImplementation(sim);
  LogComponentEnable ("FncsApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("FncsSimulatorImpl", LOG_LEVEL_INFO);

    Time::SetResolution(Time::NS);
    LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    NodeContainer nodes;
    nodes.Create(2);

    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    pointToPoint.SetChannelAttribute("Delay", StringValue("2ms"));

    NetDeviceContainer devices;
    devices = pointToPoint.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");

    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    UdpEchoServerHelper echoServer(9);

    FncsApplicationHelper fncsHelper("ns3::FncsServer", 1);

    ApplicationContainer fncsApps = fncsHelper.Install(nodes.Get(0), "node1");
    fncsApps.Add(fncsHelper.Install(nodes.Get(1), "node2"));
    fncsApps.Start(Seconds(0));
    fncsApps.Stop(Seconds(INT_MAX));

    ApplicationContainer serverApps = echoServer.Install(nodes.Get(1));
    serverApps.Start(Seconds(0));
    serverApps.Stop(Seconds(10));

    UdpEchoClientHelper echoClient(interfaces.GetAddress(1), 9);
    echoClient.SetAttribute("MaxPackets", UintegerValue(1));
    echoClient.SetAttribute("Interval", TimeValue(Seconds(1)));
    echoClient.SetAttribute("PacketSize", UintegerValue(1024));

    ApplicationContainer clientApps = echoClient.Install(nodes.Get(0));
    clientApps.Start(Seconds(0));
    clientApps.Stop(Seconds(10));

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
