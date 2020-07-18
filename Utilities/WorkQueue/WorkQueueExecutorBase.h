#pragma once

template <class InputType, class OutputType>
class WorkQueueExecutorBase {
   public:
    WorkQueueExecutorBase(){};
    virtual ~WorkQueueExecutorBase(){};

    virtual OutputType operator()(InputType &input) = 0;
};
