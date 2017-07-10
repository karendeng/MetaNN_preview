#pragma once

namespace MetaNN
{
template <>
class OperOrganizer<BinaryOpTags::Dot, CategoryTags::Matrix>
{
public:
    template <typename TD1, typename TD2>
    OperOrganizer(const TD1& data1, const TD2& data2)
        : m_rowNum(data1.RowNum())
        , m_colNum(data2.ColNum())
    {
        assert(data1.ColNum() == data2.RowNum());
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

private:
    size_t m_rowNum;
    size_t m_colNum;
};

namespace NSDot
{
template <typename TOperHandle1, typename TOperHandle2, typename TDevice>
class EvalUnit;

template <typename TOperHandle1, typename TOperHandle2>
class EvalUnit<TOperHandle1, TOperHandle2, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
    using TOperandData = std::decay_t<decltype(std::declval<TOperHandle1>().Data())>;
public:
    using ElementType = typename TOperandData::ElementType;
    using DeviceType = typename TOperandData::DeviceType;
    static_assert(std::is_same<DeviceType, DeviceTags::CPU>::value,
                  "Device type mismatch");

    EvalUnit(TOperHandle1 oper1,
             TOperHandle2 oper2,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_oper1(std::move(oper1))
        , m_oper2(std::move(oper2))
        , m_evalOutput(evalOutput) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_oper1.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        it = depMap.find(m_oper2.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& p_v1 = m_oper1.Data();
        const auto& p_v2 = m_oper2.Data();

        const size_t rowNum = p_v1.RowNum();
        const size_t colNum = p_v2.ColNum();
        const size_t midNum = p_v1.ColNum();
        assert(p_v2.RowNum() == midNum);
        
        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.Data();
        
        auto mem_res = LowerAccess(res);
        const size_t tgtPackNum = mem_res.RowLen();
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                *r = ElementType();
                for (size_t k = 0; k < midNum; ++k)
                {
                    *r += p_v1(i, k) * p_v2(k, j);
                }
                ++r;
            }
            r += tgtPackNum - colNum;
        }
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    EvalHandle<Matrix<ElementType, DeviceTags::CPU>> m_evalOutput;
};

struct GeneralCase
{
    template <typename TCaseTail, typename TEvalRes, typename TOperator1, typename TOperator2>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using RawEvalRes = typename TEvalRes::DataType;
        using DeviceType = typename RawEvalRes::DeviceType;

        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();
        using UnitType = EvalUnit<decltype(handle1), decltype(handle2), DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        UnitType unit(std::move(handle1), std::move(handle2), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }
};
}

template <>
struct OperBuildInSeq_<BinaryOpTags::Dot>
{
    using type = OperSeqContainer<NSDot::GeneralCase>;
};

template <typename TP1, typename TP2>
struct OperDot_
{
// valid check
private:
    using rawM1 = std::decay_t<TP1>;
    using rawM2 = std::decay_t<TP2>;

public:
    static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2>);

public:
    template <typename T1, typename T2,
              std::enable_if_t<std::is_same<CategoryTags::Matrix, T1>::value>* = nullptr,
              std::enable_if_t<std::is_same<CategoryTags::Matrix, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different compute types cannot dot directly");
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot dot directly");

        using ResType = BinaryOp<BinaryOpTags::Dot, rawM1, rawM2>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));

    }
};

template <typename TP1, typename TP2,
          std::enable_if_t<OperDot_<TP1, TP2>::valid>* = nullptr>
auto Dot(TP1&& p_m1, TP2&& p_m2)
{
    return OperDot_<TP1, TP2>::
            template Eval<DataCategory<TP1>, DataCategory<TP2>>(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
}
}
