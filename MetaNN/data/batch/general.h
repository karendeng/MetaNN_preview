#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/batch/facilities/tag_batch.h>
namespace MetaNN
{
template <typename TData>
class Batch
    : public TagBatch<TData, DataCategory<TData>>
{
    using TBase = TagBatch<TData, DataCategory<TData>>;
public:
    using TBase::TBase;
};
}