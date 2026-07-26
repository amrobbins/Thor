#pragma once

namespace ThorImplementation {

// Runtime capability implemented by physical layers that can switch their
// training-time dropout behavior while retaining the same placed tensors and
// learned parameters. Validation and inference remain deterministic regardless
// of this setting.
class TrainingDropoutControllable {
   public:
    virtual ~TrainingDropoutControllable() = default;

    virtual void setTrainingDropoutEnabled(bool enabled) = 0;
    [[nodiscard]] virtual bool isTrainingDropoutEnabled() const = 0;
};

}  // namespace ThorImplementation
