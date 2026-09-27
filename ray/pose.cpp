#include "ray/pose.h"


namespace ray
{


template struct Pose<float>;
template struct Pose<double>;


} // end namespace ray


template struct pex::Group
    <
        ray::PoseSchema<float>::template Schema,
        pex::PlainT<ray::Pose<float>>
    >;


template struct pex::Group
    <
        ray::PoseSchema<double>::template Schema,
        pex::PlainT<ray::Pose<double>>
    >;
