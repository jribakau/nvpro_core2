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

// Closures that free GPU resources, each run once the frame of the loop iteration it was queued
// in has completed.
//
// An iteration updates, waits for the frame that last used its ring slot, records its frame and
// submits it. What the update queues may be read by the frame in flight (the previous one, bound
// to what the update changed) and by the frame about to be recorded (an update's copy reads the
// image it replaces). Both have completed once this iteration's frame has. So a closure queued
// anywhere in an iteration joins that iteration's slot, whose queue runs when the loop is back at
// the slot, after its wait for that frame.
//
// Queuing straight into the slot would not do: the update runs before the wait and the slot's
// drain, so a closure queued in the update ran in the same iteration, with the previous frame
// still in flight. A texture moved in an update was destroyed under that frame's path trace.
class ResourceFreeRing
{
public:
  // Runs everything queued, then keeps `slots` queues (none: a closure runs as it is queued).
  void reset(uint32_t slots)
  {
    runAll();
    m_queues.assign(slots, {});
    m_held.clear();
    m_slot = kNoSlot;
  }

  void submit(std::function<void()>&& closure)
  {
    if(m_queues.empty())
      closure();
    else if(m_slot == kNoSlot)
      m_held.emplace_back(std::move(closure));  // before this iteration's drain: its slot comes then
    else
      m_queues[m_slot].emplace_back(std::move(closure));
  }

  // The loop is back at `slot` and the frame that last used it has completed: its queue runs.
  // What this iteration has queued so far, and queues until it advances, is this slot's.
  void frameCompleted(uint32_t slot)
  {
    if(slot >= m_queues.size())
      return;
    for(std::function<void()>& closure : std::exchange(m_queues[slot], {}))
      closure();
    m_queues[slot] = std::move(m_held);
    m_held.clear();
    m_slot = slot;
  }

  // The iteration's frame is submitted: what is queued from here on belongs to the next iteration.
  void frameAdvanced() { m_slot = kNoSlot; }

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
      for(std::function<void()>& closure : std::exchange(m_held, {}))
      {
        ran = true;
        closure();
      }
    }
  }

  uint32_t slots() const { return uint32_t(m_queues.size()); }

private:
  static constexpr uint32_t kNoSlot = ~0u;

  std::vector<std::vector<std::function<void()>>> m_queues;
  std::vector<std::function<void()>>              m_held;  // this iteration's, before its drain
  uint32_t                                        m_slot = kNoSlot;
};

}  // namespace nvapp
