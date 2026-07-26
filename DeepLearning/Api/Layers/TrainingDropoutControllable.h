#pragma once

#include <memory>

namespace Thor {

// Capability implemented by API layers whose stochastic training dropout can
// be enabled or disabled without changing their configured dropout rate.
//
// API layers are cloned when they are added to a Network. The shared transient
// state intentionally follows those logical clones so either a retained layer
// handle or the containing Network can set the policy used by future
// placements. The state is not part of architecture serialization.
class TrainingDropoutControllable {
   public:
    virtual ~TrainingDropoutControllable() = default;

    void setTrainingDropoutEnabled(bool enabled) { state->enabled = enabled; }
    [[nodiscard]] bool isTrainingDropoutEnabled() const { return state->enabled; }

   private:
    struct State {
        bool enabled = true;
    };

    std::shared_ptr<State> state = std::make_shared<State>();
};

}  // namespace Thor
