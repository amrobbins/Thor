#pragma once

#include "DeepLearning/Api/Loaders/LoaderBase.h"

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class Loader {
   public:
    Loader() {}
    Loader(LoaderBase *loaderBase);

    virtual ~Loader() {}

    Loader *getLoader();

   private:
    shared_ptr<LoaderBase> loader;
};

}  // namespace Thor
