#pragma once

#include <MetaNN/data/matrices/cpu_matrix.h>
#include <stdexcept>

namespace MetaNN
{
template <typename TElem>
void DataCopy(const Matrix<TElem, DeviceTags::CPU>& src,
              Matrix<TElem, DeviceTags::CPU>& dst)
{
    const size_t rowNum = src.RowNum();
    const size_t colNum = src.ColNum();
    if ((rowNum != dst.RowNum()) || (colNum != dst.ColNum()))
    {
        throw std::runtime_error("Error in data-copy: Matrix dimension mismatch.");
    }
    
    const auto mem_src = LowerAccess(src);
    const auto mem_dst = LowerAccess(dst);

    const size_t srcPackNum = mem_src.RowLen();
    const size_t dstPackNum = mem_dst.RowLen();

    using StorageType = typename Scalar<TElem, DeviceTags::CPU>::StorageType;
    const StorageType* r1 = mem_src.RawMemory();
    StorageType* r = mem_dst.MutableRawMemory();
        
    if ((srcPackNum == colNum) && (dstPackNum == colNum))
    {
        memcpy(r, r1, sizeof(StorageType) * rowNum * colNum);
    }
    else
    {
        for (size_t i = 0; i < rowNum; ++i)
        {
            memcpy(r, r1, sizeof(StorageType) * colNum);
            dst += dstPackNum;
            src += srcPackNum;
        }
    }
}
}