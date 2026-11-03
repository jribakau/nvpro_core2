/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace nvapp {

// Closures that free GPU resources, each run once every frame submitted before it was queued has
// completed.
//
// The frame loop keeps one queue per slot of its frame ring. When the loop comes back to a slot it
// has waited for the frame that last used it, and runs that slot's queue. A closure joins the queue
// of the frame submitted last, or being recorded: when that frame has completed, so has everything
// that could still read what the closure frees.
//
// Queuing it under the slot the loop is at would not do: the loop's update step runs before its
// wait, so a closure queued there ran in the same iteration, with the last frame still in flight.
// A texture moved in the update was destroyed under the previous frame's path trace that way.
class ResourceFreeRing
{
public:
  // Runs everything queued, then keeps `slots` queues (none: a closure runs as it is queued).
  void reset(uint32_t slots)
  {
    runAll();
    m_queues.assign(slots, {});
    m_slot = 0;
  }

  void submit(std::function<void()>&& closure)
  {
    if(m_queues.empty())
      closure();
    else
      m_queues[m_slot].emplace_back(std::move(closure));
  }

  // The loop is back at `slot` and the frame that last used it has completed: its queue runs.
  // What is queued from here until the loop is back at the next slot waits for the frame recorded
  // in this one.
  void frameCompleted(uint32_t slot)
  {
    if(slot >= m_queues.size())
      return;
    for(std::function<void()>& closure : std::exchange(m_queues[slot], {}))
      closure();
    m_slot = slot;
  }

  // Everything queued, including what a closure queues while running: for a device going idle.
  void runAll()
  {
    for(bool ran = true; ran;)
    {
      ran = false;
      for(std::vector<std::function<void()>>& queue : m_queues)
        for(std::function<void()>& closure : std::exchange(queue, {}))
        {
          ran = true;
          closure();
        }
    }
  }

  uint32_t slots() const { return uint32_t(m_queues.size()); }

private:
  std::vector<std::vector<std::function<void()>>> m_queues;
  uint32_t                                        m_slot = 0;
};

}  // namespace nvapp
