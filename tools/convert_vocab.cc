#include <iostream>
#include "../Thirdparty/DBoW3/src/DBoW3.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "Usage: convert_vocab input.yml output.dbow3" << std::endl;
        return 1;
    }

    std::cout << "Loading vocabulary from: " << argv[1] << " ..." << std::endl;
    DBoW3::Vocabulary voc(argv[1]);
    std::cout << "Vocabulary loaded. Words: " << voc.size() << std::endl;

    std::cout << "Saving binary to: " << argv[2] << " ..." << std::endl;
    voc.save(argv[2], true);  // true = binary format
    std::cout << "Done." << std::endl;

    return 0;
}
