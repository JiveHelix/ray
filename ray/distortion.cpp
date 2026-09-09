#include <ray/distortion.h>


namespace ray
{


namespace distortion
{


std::vector<Direction> DirectionChoices::GetChoices()
{
    return {
        Direction::forward,
        Direction::inverse};
}


std::string DirectionConverter::ToString(Direction direction)
{
    switch (direction)
    {
        case (Direction::forward):
            return "forward";

        case (Direction::inverse):
            return "inverse";

        default:
            throw std::logic_error("Unknown Direction");
    }
}


std::ostream & operator<<(std::ostream &output, Direction value)
{
    return output << DirectionConverter::ToString(value);
}


} // end namespace distortion


} // end namespace ray
