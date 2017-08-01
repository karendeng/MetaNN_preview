#pragma once

#include <MetaNN/data/batch/batch.h>
#include <MetaNN/data/batch/facilities/tag_batch.h>
#include <MetaNN/data/matrices/matrices.h>
#include <vector>

namespace MetaNN
{
template <typename TElem, typename TDevice>
class Batch<Matrix<TElem, TDevice>>
    : public TagBatch<Matrix<TElem, TDevice>, CategoryTags::Matrix>
{
    using TBase = TagBatch<Matrix<TElem, TDevice>, CategoryTags::Matrix>;
    
public:
    using TBase::TBase;

    Batch(size_t rowNum, size_t colNum, size_t batchNum)
        : TBase(rowNum, colNum)
    {
        if (batchNum == 0)
        {
            return;
        }

        for (size_t i = 0; i < batchNum; ++i)
        {
            TBase::EmplaceBack(rowNum, colNum);
        }
    }
    
    auto SubMatrix(size_t p_rowB, size_t p_rowE, size_t p_colB, size_t p_colE) const
    {
        Batch<Matrix<TElem, TDevice>> res(p_rowE - p_rowB, p_colE - p_colB);
        res.Reserve(TBase::BatchNum());
        for (auto it = TBase::begin(); it != TBase::end(); ++it)
        {
            res.PushBack(it->SubMatrix(p_rowB, p_rowE, p_colB, p_colE));
        }
        return res;
    }
};
}