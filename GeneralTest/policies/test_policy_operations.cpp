#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
struct Tag1;
struct Tag2;
struct Tag3;

void test_policy_operations1()
{
    cout << "Test policy operations case 1...\t";
    using input = PolicyContainer<PRowVecInput, PFloatElement,
                                  SubPolicyContainer<Tag1, PColVecInput,
                                                     SubPolicyContainer<Tag2>>>;
    using check1 = SubPolicyPicker<input, Tag3>;
    static_assert(is_same<check1, PolicyContainer<PRowVecInput, PFloatElement>>::value, "Check Error");

    using check2 = SubPolicyPicker<input, Tag1>;
    static_assert(is_same<check2, PolicyContainer<PColVecInput, SubPolicyContainer<Tag2>, PFloatElement>>::value, "Check Error");

    using check3 = SubPolicyPicker<check2, Tag3>;
    static_assert(is_same<check3, PolicyContainer<PColVecInput, PFloatElement>>::value, "Check Error");

    using check4 = SubPolicyPicker<check2, Tag2>;
    static_assert(is_same<check4, PolicyContainer<PColVecInput, PFloatElement>>::value, "Check Error");
    cout << "done" << endl;
}
}
void test_policy_operations()
{
    test_policy_operations1();
}
