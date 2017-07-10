#pragma once
#include <unordered_map>

namespace MetaNN
{
template <typename TElement, typename TDevice>
struct MatrixGradInfo
{
    using GradItemType = DynamicType<Matrix<TElement, TDevice>>;

    MatrixGradInfo(Matrix<TElement, TDevice> p_weight)
        : weight(std::move(p_weight))
        , grad(p_weight.RowNum(), p_weight.ColNum()) {}

    Matrix<TElement, TDevice> weight;
    Batch<GradItemType> grad;
};

template <typename TElement, typename TDevice>
class GradCollectorIterator
{
    using IteratorType = typename std::unordered_map<const TElement*, MatrixGradInfo<TElement, TDevice>>::iterator;

public:
    GradCollectorIterator(IteratorType it)
        : m_it(std::move(it)) {}

    const auto& operator* () const
    {
        return m_it->second;
    }

    const auto operator-> () const
    {
        return &(m_it->second);
    }

    auto operator++()
    {
        ++m_it;
        return *this;
    }

    auto operator++(int)
    {
        auto tmp = *this;
        ++m_it;
        return tmp;
    }

    bool operator== (const GradCollectorIterator& git) const
    {
        return m_it == git.m_it;
    }

    bool operator!= (const GradCollectorIterator& git) const
    {
        return !(operator==(git));
    }

private:
    IteratorType m_it;
};

template <typename TElement, typename TDevice>
class GradCollector
{
public:
    GradCollector() = default;
    GradCollector(const GradCollector&) = delete;
    GradCollector(GradCollector&&) = default;
    GradCollector& operator = (const GradCollector&) = delete;
    GradCollector& operator = (GradCollector&&) = delete;

    template<typename TGrad>
    void Collect(const Matrix<TElement, TDevice>& weight,
                 const TGrad& grad)
    {
        auto mem = LowerAccess(weight);
        auto buf = mem.RawMemory();

        auto it = m_matricesInfo.find(buf);

        if (it != m_matricesInfo.end())
        {
            it->second.grad.PushBack(MakeDynamic(grad));
        }
        else
        {
            MatrixGradInfo<TElement, TDevice> mgi(weight);
            mgi.grad.PushBack(MakeDynamic(grad));

            m_matricesInfo.insert({buf, std::move(mgi)});
        }
    }

    void clear()
    {
        m_matricesInfo.clear();
    }

    size_t size() const
    {
        return m_matricesInfo.size();
    }

    auto begin()
    {
        return GradCollectorIterator<TElement, TDevice>(m_matricesInfo.begin());
    }

    auto begin() const
    {
        return GradCollectorIterator<TElement, TDevice>(m_matricesInfo.begin());
    }

    auto end()
    {
        return GradCollectorIterator<TElement, TDevice>(m_matricesInfo.end());
    }

    auto end() const
    {
        return GradCollectorIterator<TElement, TDevice>(m_matricesInfo.end());
    }

private:
    std::unordered_map<const TElement*, MatrixGradInfo<TElement, TDevice>> m_matricesInfo;
};
}
