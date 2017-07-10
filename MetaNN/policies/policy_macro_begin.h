#define EnumPolicyObj(PolicyName, Ma, Mi, Val) \
struct PolicyName : virtual public Ma\
{ \
    using MajorClass = Ma; \
    using MinorClass = Ma::Mi##Enum; \
    using Mi = Ma::Mi##Enum::Val; \
}

#define ValuePolicyObj(PolicyName, Ma, Mi, Val) \
struct PolicyName : virtual public Ma \
{ \
    using MajorClass = Ma; \
    using MinorClass = Ma::Mi##Value; \
private: \
    using type1 = decltype(Ma::Mi); \
    using type2 = std::decay_t<type1>; \
public: \
    static constexpr type2 Mi = static_cast<type2>(Val); \
}

#define TypePolicyObj(PolicyName, Ma, Mi, Val) \
struct PolicyName : virtual public Ma \
{ \
    using MajorClass = Ma; \
    using MinorClass = Ma::Mi##Type; \
    using Mi = Val; \
}
