/*
 * Strict batch adapter for the original SZ 1.1 C API.
 *
 * The compression implementation is not reproduced here. This program calls
 * the upstream SZ_compress_args/SZ_decompress functions from commit
 * 2123f4899b76e6dfe28861da0a70d63a6cd43f83 of szcompressor/sz2.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sz.h"
#include "zlib.h"

static void fail(const char *message)
{
    fprintf(stderr, "sz2016_batch: %s\n", message);
    exit(EXIT_FAILURE);
}

static FILE *open_file(const char *path, const char *mode)
{
    FILE *file = fopen(path, mode);
    if (file == NULL) {
        fprintf(stderr, "sz2016_batch: cannot open %s (errno=%d)\n", path, errno);
        exit(EXIT_FAILURE);
    }
    return file;
}

static long parse_positive(const char *text, const char *name)
{
    char *end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value <= 0) {
        fprintf(stderr, "sz2016_batch: invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return value;
}

static size_t checked_product(size_t left, size_t right, const char *label)
{
    if (left != 0 && right > SIZE_MAX / left) {
        fprintf(stderr, "sz2016_batch: %s overflows size_t\n", label);
        exit(EXIT_FAILURE);
    }
    return left * right;
}

static void write_u32_le(FILE *file, uint32_t value)
{
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    bytes[2] = (unsigned char)((value >> 16) & 0xffu);
    bytes[3] = (unsigned char)((value >> 24) & 0xffu);
    if (fwrite(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        fail("cannot write framed stream length");
    }
}

static uint32_t read_u32_le(FILE *file)
{
    unsigned char bytes[4];
    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        fail("truncated framed SZ stream length");
    }
    return (uint32_t)bytes[0]
        | ((uint32_t)bytes[1] << 8)
        | ((uint32_t)bytes[2] << 16)
        | ((uint32_t)bytes[3] << 24);
}

/* Require the bundled zlib member to consume exactly one non-within-range frame. */
static void validate_official_stream(
    const char *compressed,
    int compressed_size,
    size_t maximum_flat_bytes
)
{
    z_stream stream;
    char *flat_bytes;
    int status;

    if (compressed_size == 12) {
        return;
    }
    flat_bytes = (char *)malloc(maximum_flat_bytes);
    if (flat_bytes == NULL) {
        fail("cannot allocate the SZ zlib validation buffer");
    }
    memset(&stream, 0, sizeof(stream));
    stream.next_in = (Bytef *)compressed;
    stream.avail_in = (uInt)compressed_size;
    stream.next_out = (Bytef *)flat_bytes;
    stream.avail_out = (uInt)maximum_flat_bytes;
    status = inflateInit(&stream);
    if (status != Z_OK) {
        free(flat_bytes);
        fail("cannot initialize validation for an official SZ zlib stream");
    }
    status = inflate(&stream, Z_FINISH);
    if (status != Z_STREAM_END || stream.avail_in != 0) {
        (void)inflateEnd(&stream);
        free(flat_bytes);
        fail("truncated or trailing bytes inside an official SZ window payload");
    }
    if (inflateEnd(&stream) != Z_OK) {
        free(flat_bytes);
        fail("cannot finalize validation for an official SZ zlib stream");
    }
    free(flat_bytes);
}

/*
 * SZ 1.1's float decompressor aliases a 12-byte within-range input as its
 * temporary buffer and then frees that buffer. The caller still owns the
 * persisted official bytes, so give upstream an equally sized malloc copy in
 * exactly that case. For every other stream SZ allocates and frees its own
 * zlib output and does not consume the caller's compressed buffer.
 */
static float *decompress_official(
    const char *compressed,
    int compressed_size,
    int channels,
    int time_points
)
{
    char *upstream_input = (char *)compressed;
    int second_dimension = channels;

    if (compressed_size == 12) {
        upstream_input = (char *)malloc(12);
        if (upstream_input == NULL) {
            fail("cannot allocate the 12-byte SZ decompression scratch copy");
        }
        memcpy(upstream_input, compressed, 12);
        /* SZ_decompress consumes this scratch allocation at the pinned commit. */
    }
    if (channels == 1) {
        second_dimension = 0;
    }
    return (float *)SZ_decompress(
        SZ_FLOAT,
        upstream_input,
        compressed_size,
        0,
        0,
        0,
        second_dimension,
        time_points
    );
}

static double decode_persisted_stream(
    const char *stream_path,
    const char *reconstruction_path,
    long batch,
    int channels,
    int time_points,
    const float *source,
    unsigned long long *native_bytes
)
{
    FILE *streams = open_file(stream_path, "rb");
    FILE *reconstruction = open_file(reconstruction_path, "wb");
    size_t values_per_window = checked_product(
        (size_t)channels,
        (size_t)time_points,
        "values per window"
    );
    size_t maximum_flat_bytes = checked_product(
        values_per_window,
        sizeof(float),
        "SZ flat-byte validation size"
    );
    double max_abs_error = 0.0;
    long index;

    if (native_bytes != NULL) {
        *native_bytes = 0;
    }
    for (index = 0; index < batch; ++index) {
        uint32_t framed_size = read_u32_le(streams);
        char *compressed;
        float *decoded;
        size_t value_index;

        if (framed_size == 0 || framed_size > (uint32_t)INT_MAX) {
            fail("invalid framed SZ window length");
        }
        compressed = (char *)malloc((size_t)framed_size);
        if (compressed == NULL) {
            fail("cannot allocate framed SZ window");
        }
        if (fread(compressed, 1, (size_t)framed_size, streams) != (size_t)framed_size) {
            free(compressed);
            fail("truncated framed SZ window payload");
        }

        validate_official_stream(compressed, (int)framed_size, maximum_flat_bytes);

        decoded = decompress_official(
            compressed,
            (int)framed_size,
            channels,
            time_points
        );
        if (decoded == NULL) {
            free(compressed);
            fail("upstream SZ decompression failed");
        }
        for (value_index = 0; value_index < values_per_window; ++value_index) {
            if (!isfinite((double)decoded[value_index])) {
                free(compressed);
                free(decoded);
                fail("upstream SZ decoded NaN or infinity");
            }
            if (source != NULL) {
                size_t source_index = (size_t)index * values_per_window + value_index;
                double error = fabs(
                    (double)source[source_index] - (double)decoded[value_index]
                );
                if (error > max_abs_error) {
                    max_abs_error = error;
                }
            }
        }
        if (fwrite(decoded, sizeof(float), values_per_window, reconstruction)
            != values_per_window) {
            free(compressed);
            free(decoded);
            fail("cannot write reconstructed float32 tensor");
        }
        if (native_bytes != NULL) {
            *native_bytes += (unsigned long long)framed_size;
        }
        free(compressed);
        free(decoded);
    }
    if (fgetc(streams) != EOF) {
        fail("framed SZ stream has trailing bytes");
    }
    if (ferror(streams)) {
        fail("cannot finish reading framed SZ stream");
    }
    if (fclose(streams) != 0 || fclose(reconstruction) != 0) {
        fail("cannot close persisted SZ replay files");
    }
    return max_abs_error;
}

static int decode_command(int argc, char **argv)
{
    const char *config_path;
    const char *stream_path;
    const char *reconstruction_path;
    long batch;
    long channels;
    long time_points;

    if (argc != 8) {
        return 0;
    }
    config_path = argv[2];
    stream_path = argv[3];
    reconstruction_path = argv[4];
    batch = parse_positive(argv[5], "batch size");
    channels = parse_positive(argv[6], "channel count");
    time_points = parse_positive(argv[7], "time length");
    if (channels > INT_MAX || time_points > INT_MAX) {
        fail("window dimensions exceed the SZ 1.1 integer API");
    }
    if (checked_product((size_t)channels, (size_t)time_points, "window shape")
        > (size_t)INT_MAX / sizeof(float)) {
        fail("window is too large for the SZ 1.1 signed byte-count arithmetic");
    }

    /* The pinned upstream declares int but has no return statement on success. */
    (void)SZ_Init((char *)config_path);
    (void)decode_persisted_stream(
        stream_path,
        reconstruction_path,
        batch,
        (int)channels,
        (int)time_points,
        NULL,
        NULL
    );
    SZ_Finalize();
    return 1;
}

int main(int argc, char **argv)
{
    const char *config_path;
    const char *input_path;
    const char *stream_path;
    const char *reconstruction_path;
    long batch;
    long channels;
    long time_points;
    size_t values_per_window;
    size_t total_values;
    double epsilon;
    char *epsilon_end = NULL;
    float *source;
    FILE *input;
    FILE *streams;
    long index;
    double max_abs_error;
    unsigned long long native_bytes = 0;

    if (argc >= 2 && strcmp(argv[1], "--decode") == 0) {
        if (!decode_command(argc, argv)) {
            fprintf(
                stderr,
                "usage: %s --decode CONFIG STREAMS.sz RECON.f32 BATCH CHANNELS TIME\n",
                argv[0]
            );
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    if (argc != 9) {
        fprintf(
            stderr,
            "usage: %s CONFIG INPUT.f32 STREAMS.sz RECON.f32 BATCH CHANNELS TIME EPSILON\n"
            "       %s --decode CONFIG STREAMS.sz RECON.f32 BATCH CHANNELS TIME\n",
            argv[0],
            argv[0]
        );
        return EXIT_FAILURE;
    }

    config_path = argv[1];
    input_path = argv[2];
    stream_path = argv[3];
    reconstruction_path = argv[4];
    batch = parse_positive(argv[5], "batch size");
    channels = parse_positive(argv[6], "channel count");
    time_points = parse_positive(argv[7], "time length");
    errno = 0;
    epsilon = strtod(argv[8], &epsilon_end);
    if (errno == ERANGE || epsilon_end == argv[8] || *epsilon_end != '\0'
        || !isfinite(epsilon) || epsilon <= 0.0) {
        fail("epsilon must be a positive finite number");
    }
    if (channels > INT_MAX || time_points > INT_MAX) {
        fail("window dimensions exceed the SZ 1.1 integer API");
    }

    values_per_window = checked_product(
        (size_t)channels,
        (size_t)time_points,
        "values per window"
    );
    total_values = checked_product((size_t)batch, values_per_window, "tensor shape");
    (void)checked_product(total_values, sizeof(float), "input byte count");
    if (values_per_window > (size_t)INT_MAX / sizeof(float)) {
        fail("window is too large for the SZ 1.1 signed byte-count arithmetic");
    }
    source = (float *)malloc(total_values * sizeof(float));
    if (source == NULL) {
        fail("cannot allocate input array");
    }

    input = open_file(input_path, "rb");
    if (fread(source, sizeof(float), total_values, input) != total_values) {
        fail("input file does not contain the requested float32 tensor");
    }
    if (fgetc(input) != EOF) {
        fail("input file contains trailing bytes");
    }
    if (ferror(input) || fclose(input) != 0) {
        fail("cannot finish reading the input tensor");
    }
    {
        size_t value_index;
        for (value_index = 0; value_index < total_values; ++value_index) {
            if (!isfinite((double)source[value_index])) {
                fail("input tensor contains NaN or infinity");
            }
        }
    }

    streams = open_file(stream_path, "wb");
    /* The pinned upstream declares int but has no return statement on success. */
    (void)SZ_Init((char *)config_path);

    for (index = 0; index < batch; ++index) {
        float *window = source + (size_t)index * values_per_window;
        int compressed_size = 0;
        int second_dimension = channels == 1 ? 0 : (int)channels;
        char *compressed = SZ_compress_args(
            SZ_FLOAT,
            window,
            &compressed_size,
            ABS,
            epsilon,
            0.0,
            0,
            0,
            0,
            second_dimension,
            (int)time_points
        );

        if (compressed == NULL || compressed_size <= 0) {
            free(compressed);
            fail("upstream SZ compression failed");
        }
        write_u32_le(streams, (uint32_t)compressed_size);
        if (fwrite(compressed, 1, (size_t)compressed_size, streams)
            != (size_t)compressed_size) {
            free(compressed);
            fail("cannot write native SZ stream");
        }
        free(compressed);
    }
    if (fclose(streams) != 0) {
        fail("cannot close the persisted SZ stream");
    }

    /* Benchmark reconstruction is decoded from the exact bytes charged to the stream. */
    max_abs_error = decode_persisted_stream(
        stream_path,
        reconstruction_path,
        batch,
        (int)channels,
        (int)time_points,
        source,
        &native_bytes
    );
    SZ_Finalize();
    free(source);

    if (max_abs_error > epsilon + 1e-5) {
        fprintf(
            stderr,
            "sz2016_batch: upstream SZ violated the error bound: %.9g > %.9g\n",
            max_abs_error,
            epsilon
        );
        return EXIT_FAILURE;
    }
    printf(
        "windows=%ld native_bytes=%llu max_abs_error=%.9g\n",
        batch,
        native_bytes,
        max_abs_error
    );
    return EXIT_SUCCESS;
}
