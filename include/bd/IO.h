#ifndef __BD_IO_h
#define __BD_IO_h

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <limits>  


namespace {
  std::vector< std::string > 
  split(std::string s, char delim) {
    std::vector< std::string > tokens;
    std::size_t start = 0;
    for (std::size_t end = start; end < s.size(); ++end) {
      if (s[end] == delim) {
	tokens.push_back(s.substr(start, end-start));
	start = end + 1;
      }
    }
    if (start < s.size()) {
      tokens.push_back(s.substr(start, s.npos));
    }
    return tokens;
  }

}



template<typename CharT, typename ElemT>
ElemT
parseElementFromString(const std::basic_string<CharT>& s) {
  ElemT e;
  std::basic_istringstream< CharT >(s) >> e;
  return e;
}

template< typename ElemT, typename CharT, typename OutputIt >
std::pair<size_t, size_t>
readTextMatrix( std::istream& is,
		OutputIt out,
		CharT colSep=',',
		CharT rowSep='\n' ) {
  size_t rows = 0;
  size_t cols = 0;
  std::basic_string< CharT > row;
  while ( is.good() ) {
    // Read a row and split into columns
    std::getline( is, row, rowSep );
    auto elements = split( row, colSep );
    if ( elements.size() > 0 ) {
      // We use the number of elements in the first row as the number of columns
      // and assert that following rows have the same sumber of columns
      if ( rows == 0 ) {
	cols = elements.size();
      }
      assert( elements.size() == cols );
      
      // Now we iterate over the elements, convert to the proper type and store
      // in out
      for ( const auto& element : elements ) {
	*out++ = parseElementFromString<CharT,ElemT>(element);
      }
      ++rows;
    }
  }
  return std::make_pair(rows, cols);
}

#endif
