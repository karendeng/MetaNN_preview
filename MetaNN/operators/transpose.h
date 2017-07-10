#pragma once

namespace MetaNN
{
template <>
class OperOrganizer<UnaryOpTags::Transpose, CategoryTags::Matrix>
{
public:
    template <typename TData>
    OperOrganizer(const TData& data)
        : m_rowNum(data.ColNum())
        , m_colNum(data.RowNum())
    { }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

private:
    size_t m_rowNum;
    size_t m_colNum;
};

namespace NSTranspose
{
template <typename TOperHandle, typename TDevice>
class EvalUnit;

template <typename TOperHandle>
class EvalUnit<TOperHandle, DeviceTags::CPU> : public BaseEvalUnit<DeviceTags::CPU>
{
    using TOperandData = std::decay_t<decltype(std::declval<TOperHandle>().Data())>;
public:
    using ElementType = typename TOperandData::ElementType;
    using DeviceType = typename TOperandData::DeviceType;
    static_assert(std::is_same<DeviceType, DeviceTags::CPU>::value,
                  "Device type mismatch");

    EvalUnit(TOperHandle oper,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_oper(std::move(oper))
        , m_evalOutput(evalOutput) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_oper.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& p_v = m_oper.Data();
        const size_t rowNum = p_v.RowNum();
        const size_t colNum = p_v.ColNum();
        
        m_evalOutput.Allocate(colNum, rowNum);
        auto& res = m_evalOutput.Data();

        auto mem_v1 = LowerAccess(p_v);
        const size_t src1PackNum = mem_v1.RowLen();
        const ElementType* r1 = mem_v1.RawMemory();

        auto mem_res = LowerAccess(res);
        const size_t resPackNum = mem_res.RowLen();
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j * resPackNum + i] = r1[j];
            }
            r1 += src1PackNum;
        }
    }

private:
    TOperHandle m_oper;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct GeneralCase
{
    template <typename TCaseTail, typename TEvalRes, typename TOperand>
    static void EvalRegister(TEvalRes& evalRes, const TOperand& oper)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using RawEvalRes = typename TEvalRes::DataType;
        using DeviceType = typename RawEvalRes::DeviceType;

        auto handle = oper.EvalRegister();
        using UnitType = EvalUnit<decltype(handle), DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        UnitType unit(std::move(handle), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }
};
}

template <>
struct OperBuildInSeq_<UnaryOpTags::Transpose>
{
    using type = OperSeqContainer<NSTranspose::GeneralCase>;
};

template <typename TP>
struct OperTranspose_
{
// valid check
private:
    using rawM = std::decay_t<TP>;

public:
    static constexpr bool valid = IsMatrix<rawM>;

public:
    static auto Eval(TP&& p_m)
    {
        using ResType = UnaryOp<UnaryOpTags::Transpose, rawM>;
        return ResType(std::forward<TP>(p_m));
    }
};

template <typename TP,
          std::enable_if_t<OperTranspose_<TP>::valid>* = nullptr>
auto Transpose(TP&& p_m)
{
    return OperTranspose_<TP>::Eval(std::forward<TP>(p_m));
}
}
