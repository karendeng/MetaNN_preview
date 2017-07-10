#pragma once

#include <MetaNN/facilities/named_params.h>

namespace MetaNN
{
struct LayerIO : public NamedParams<LayerIO> {};

struct CostLayerIn : public NamedParams<CostLayerIn, struct CostLayerLabel> {};

struct RnnLayerHiddenBefore;
}
