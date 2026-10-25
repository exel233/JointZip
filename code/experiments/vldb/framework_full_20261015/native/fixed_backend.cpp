// Thin transport over pinned official APIs. SPERR uses the historical
// native baseline SPECK1D_FLT (not the portfolio's 2D mode); Serf matches
// the historical guided experiment. No search, repair, or algorithm changes.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "SPERR_C_API.h"
#include "SPECK1D_FLT.h"
#include "Conditioner.h"
#include <cstring>
#include "compressor_32/serf_qt_compressor_32.h"
#include "compressor_32/serf_xor_compressor_32.h"
#include "decompressor_32/serf_qt_decompressor_32.h"
#include "decompressor_32/serf_xor_decompressor_32.h"

template<class T> std::vector<T> read(const char* path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open input");
  auto n=in.tellg();
  if (n<=0 || static_cast<size_t>(n)%sizeof(T)) throw std::runtime_error("bad input length");
  std::vector<T> result(static_cast<size_t>(n)/sizeof(T)); in.seekg(0);
  if (!in.read(reinterpret_cast<char*>(result.data()),n)) throw std::runtime_error("input read failed");
  return result;
}
template<class T> void write(const char* path,const std::vector<T>& values) {
  std::ofstream out(path,std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
  if (!out) throw std::runtime_error("output write failed");
}
// Retain the complete-stream check from the historical SPERR baseline.
void validate_sperr(const std::vector<unsigned char>& bytes) {
  if(bytes.size()<17) throw std::runtime_error("truncated SPERR conditioner");
  if(sperr::Conditioner{}.is_constant(bytes[0])) {
    if(bytes.size()!=17) throw std::runtime_error("trailing SPERR constant bytes");
    return;
  }
  auto section=[&](size_t offset) {
    if(bytes.size()-offset<9 || bytes[offset]>64) throw std::runtime_error("bad SPECK header");
    uint64_t bits=0;std::memcpy(&bits,bytes.data()+offset+1,8);
    uint64_t size=9+bits/8+(bits%8!=0);
    if(size>bytes.size()-offset) throw std::runtime_error("truncated SPECK body");
    return offset+static_cast<size_t>(size);
  };
  auto end=section(17);
  if(end!=bytes.size() && section(end)!=bytes.size()) throw std::runtime_error("trailing SPECK bytes");
}
int main(int argc,char** argv) {
  try {
    if (argc!=7) throw std::runtime_error("fixed_backend c|d sperr|serf_xor|serf_qt input output length cap");
    const std::string operation=argv[1],codec=argv[2];
    const size_t n=std::stoull(argv[5]); const double cap=std::stod(argv[6]);
    if (!n || n>static_cast<size_t>(std::numeric_limits<int>::max()) || !(cap>0) || !std::isfinite(cap))
      throw std::runtime_error("invalid size or cap");
    if (operation=="c") {
      auto values=read<float>(argv[3]);
      if (values.size()!=n || !std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}))
        throw std::runtime_error("invalid float source");
      std::vector<unsigned char> output;
      if(codec=="sperr") {
        auto compressor=sperr::SPECK1D_FLT{};
        compressor.set_dims({n,1,1});compressor.copy_data(values.data(),n);
        compressor.set_tolerance(cap);
        if(compressor.compress()!=sperr::RTNType::Good) throw std::runtime_error("SPERR-1D encode failed");
        compressor.append_encoded_bitstream(output);validate_sperr(output);
      } else {
        if(cap>std::numeric_limits<float>::max()) throw std::runtime_error("Serf cap overflow");
        Array<std::uint8_t> encoded;
        if(codec=="serf_xor") {
          auto compressor=SerfXORCompressor32(static_cast<int>(n),static_cast<float>(cap));
          for(float v:values) compressor.AddValue(v); compressor.Close();
          encoded=compressor.compressed_bytes_last_block();
        } else if(codec=="serf_qt") {
          auto compressor=SerfQtCompressor32(static_cast<int>(n),static_cast<float>(cap));
          for(float v:values) compressor.AddValue(v); compressor.Close();
          encoded=compressor.compressed_bytes();
        } else throw std::runtime_error("unsupported codec");
        output.assign(encoded.begin(),encoded.end());
      }
      if(output.empty()) throw std::runtime_error("empty native stream");
      write(argv[4],output);
    } else if(operation=="d") {
      auto input=read<unsigned char>(argv[3]);std::vector<float> output;
      if(codec=="sperr") {
        validate_sperr(input);auto decoder=sperr::SPECK1D_FLT{};
        decoder.set_dims({n,1,1});
        if(decoder.use_bitstream(input.data(),input.size())!=sperr::RTNType::Good ||
           decoder.decompress()!=sperr::RTNType::Good) throw std::runtime_error("SPERR-1D decode failed");
        auto values=decoder.release_decoded_data();
        output.resize(values.size());std::transform(values.begin(),values.end(),output.begin(),[](double v){return static_cast<float>(v);});
      } else {
        const auto encoded=Array<std::uint8_t>(std::vector<std::uint8_t>(input.begin(),input.end()));
        if(codec=="serf_xor") output=SerfXORDecompressor32().Decompress(encoded);
        else if(codec=="serf_qt") output=SerfQtDecompressor32().Decompress(encoded);
        else throw std::runtime_error("unsupported codec");
      }
      if(output.size()!=n || !std::all_of(output.begin(),output.end(),[](float v){return std::isfinite(v);}))
        throw std::runtime_error("invalid reconstruction");
      write(argv[4],output);
    } else throw std::runtime_error("unsupported operation");
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
