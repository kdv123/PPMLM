// Home to various constants used by the other PPM-related classes

enum Constants
{
    static let ROOT_ID = 0              // Numeric ID for root of PPM tree
    static let ROOT_NAME = "<R>"        // Name of the root node
    static let KN_ALPHA = 0.49          // Copied from Dasher, used in probability calculation
    static let KN_BEAT = 0.77           // Copied from Dasher, used in probability calculation
    static let EPSILON = 1e-10          // Small value for comparing floating point values
}
