#pragma once

#include <boost/filesystem.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/set.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>

typedef boost::interprocess::allocator<uint8_t, boost::interprocess::managed_mapped_file::segment_manager> file_vector_allocator_t;
typedef boost::interprocess::vector<uint8_t, file_vector_allocator_t> file_vector_t;

typedef boost::interprocess::allocator<char, boost::interprocess::managed_mapped_file::segment_manager> file_string_allocator_t;
typedef boost::interprocess::basic_string<char, std::char_traits<char>, file_string_allocator_t> file_string_t;

typedef boost::interprocess::allocator<file_string_t, boost::interprocess::managed_mapped_file::segment_manager>
    file_string_vector_allocator_t;
typedef boost::interprocess::vector<file_string_t, file_string_vector_allocator_t> file_string_vector_t;
