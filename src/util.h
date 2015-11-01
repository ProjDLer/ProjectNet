#pragma once
typedef float real;

#include <cstdlib>
#include <random>
#include <map>
#include "Dictionary.h"
#include "HuffmanEncoder.h"
#include <algorithm>

#define MAX_RELAT_CNT 50
#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000
#define MAX_EXP 12
#define MAX_BATCH_SIZE 105
#define MAX_EMBEDDING_SIZE 105
#define MAX_RELAT_RANK 100
#define MIN_LOG -15

const int table_size = (int)1e8;
const double eps = 1e-6;
const real log_ratio = 200.0 / 100;

enum class UpdateDirection
{
	UPDATE_INPUT,
	UPDATE_OUTPUT
};

struct Opt
{
	static const char* train_file;
	static const char* save_vocab_file;
	static const char* read_vocab_file;
	static int output_binary;
	static const char* binary_embedding_file;
	static const char* text_embedding_file;
	static const char* huff_tree_file;
	static const char* outputlayer_binary_file;
	static const char* outputlayer_text_file;
	static const char* sw_file;

	static const char* relation_file;
	static bool use_relation;
	static const char* out_relat_text;
	static const char* out_relat_binary;

	static bool hs, stopwords;
	static bool update_mat;
	static bool use_know_in_text;
	static bool act_relat;
	static bool sig_relat;
	static bool use_tail_mat; //Specify whether we use the projection mat for the tail entity
	static real sample;
	static int batch_size, embeding_size, thread_cnt, window_size, negative_num, min_count, epoch;
	static int know_iter;
	static int head_relat_rank;
	static int tail_relat_rank;
	static bool act_relat_mat;
	static real init_learning_rate;
	static real lambda; //specify (1-\lambda)*L_corpus + \lambda*loglikelyhood(h-t)
	static real margin; //specify the margin
	static real relat_neg_weight;
	static real know_update_per_progress;
	static bool is_diag;

	static void ParseArgs(int argc, char* argv[])
	{
		for (int i = 1; i < argc; i += 2)
		{
			if (strcmp(argv[i], "-size") == 0) embeding_size = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-train") == 0) train_file = argv[i + 1];
			if (strcmp(argv[i], "-save_vocab") == 0) save_vocab_file = argv[i + 1];
			if (strcmp(argv[i], "-read_vocab") == 0) read_vocab_file = argv[i + 1];
			if (strcmp(argv[i], "-binary") == 0) output_binary = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-init_learning_rate") == 0) init_learning_rate = (real)atof(argv[i + 1]);
			if (strcmp(argv[i], "-binary_embedding_file") == 0) binary_embedding_file = argv[i + 1];
			if (strcmp(argv[i], "-text_embedding_file") == 0) text_embedding_file = argv[i + 1];
			if (strcmp(argv[i], "-window") == 0) window_size = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-sample") == 0) sample = (real)atof(argv[i + 1]);
			if (strcmp(argv[i], "-hs") == 0) hs = atoi(argv[i + 1]) != 0;
			if (strcmp(argv[i], "-negative") == 0) negative_num = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-threads") == 0) thread_cnt = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-min_count") == 0) min_count = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-epoch") == 0) epoch = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-stopwords") == 0) stopwords = atoi(argv[i + 1]) != 0;
			if (strcmp(argv[i], "-sw_file") == 0) sw_file = argv[i + 1];

			if (strcmp(argv[i], "-outputlayer_text_file") == 0) outputlayer_text_file = argv[i + 1];
			if (strcmp(argv[i], "-outputlayer_binary_file") == 0) outputlayer_binary_file = argv[i + 1];
			if (strcmp(argv[i], "-out_relat_text") == 0) out_relat_text = argv[i + 1];
			if (strcmp(argv[i], "-out_relat_binary") == 0) out_relat_binary = argv[i + 1];

			if (strcmp(argv[i], "-batch_size") == 0) batch_size = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-know_iter") == 0) know_iter = atoi(argv[i + 1]);

			if (strcmp(argv[i], "-update_mat") == 0) update_mat = atoi(argv[i + 1])  !=  0;
			if (strcmp(argv[i], "-use_tail_mat") == 0) use_tail_mat = atoi(argv[i + 1]) != 0;
			if (strcmp(argv[i], "-know_in_text") == 0) use_know_in_text = atoi(argv[i + 1]) != 0;
			if (strcmp(argv[i], "-lambda") == 0) lambda = (real)atof(argv[i + 1]);
			if (strcmp(argv[i], "-margin") == 0) margin = (real)atof(argv[i + 1]);
			if (strcmp(argv[i], "-relation_file") == 0) relation_file = argv[i + 1];
			if (strcmp(argv[i], "-use_relation") == 0) use_relation = atoi(argv[i + 1]) !=  0;
			if (strcmp(argv[i], "-head_relat_rank") == 0) head_relat_rank = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-tail_relat_rank") == 0) tail_relat_rank = atoi(argv[i + 1]);
			if (strcmp(argv[i], "-act_relat") == 0) act_relat = atoi(argv[i + 1])  != 0;
			if (strcmp(argv[i], "-act_relat_mat") == 0) act_relat_mat = atoi(argv[i + 1]) !=  0;
			if (strcmp(argv[i], "-sig_relat") == 0) sig_relat = atoi(argv[i + 1]) != 0;
			if (strcmp(argv[i], "-relat_neg_weight") == 0) relat_neg_weight = (real)atof(argv[i + 1]);
			if (strcmp(argv[i], "-update_ratio") == 0) know_update_per_progress = (real)atof(argv[i + 1]);
			if (strcmp(argv[i], "-is_diag") == 0) is_diag = atoi(argv[i + 1])!= 0;

		}
		use_know_in_text &= use_relation;
		
		printf("Text: %s, alpha: %.5f, -thread: %d, know_iter: %d, head_relat_rank: %d tail_relat_rank: %d labmbda: %.4f use_tail_mat:%d, act_relat:%d, update ratio:%.4f, update_mat: %d, is_diag:%d\n", train_file, init_learning_rate, thread_cnt, know_iter, 
			head_relat_rank, tail_relat_rank, lambda, use_tail_mat, act_relat, know_update_per_progress, update_mat, is_diag);
	}
};

class Sampler
{
public:
	static void SetNegativeSamplingDistribution(Dictionary* dict)
	{
		long long train_words_pow = 0;
		real power = 0.75;
		table = (int *)malloc(table_size * sizeof(int));
		for (int i = 0; i < dict->Size(); ++i)
			train_words_pow += (long long)pow(dict->GetWordInfo(i)->freq, power);
		int cur_pos = 0;
		real d1 = (real)pow(dict->GetWordInfo(cur_pos)->freq, power) / (real)train_words_pow;

		for (int i = 0; i < table_size; ++i) 
		{
			table[i] = cur_pos;
			if (i / (real)table_size > d1) 
			{
				cur_pos++;
				d1 += (real)pow(dict->GetWordInfo(cur_pos)->freq, power) / (real)train_words_pow;
			}
			if (cur_pos >= dict->Size()) 
				cur_pos = dict->Size() - 1;
		}
	}

	static bool WordSampling(long long word_cnt, long long train_words)
	{
		real ran = (sqrt(word_cnt / (Opt::sample * train_words)) + 1) * (Opt::sample * train_words) / word_cnt;
		return (ran > ((real) rand() / (RAND_MAX)));
	}

	static int NegativeSampling()
	{
		return table[(int_distribution)(generator)];
	}

private:
	static int* table;
	static std::default_random_engine generator;
	static std::uniform_int_distribution<int> int_distribution;
};

class Util
{
public:
	static bool ReadWord(char *word, FILE *fin);
	static void SaveVocab();

	template<typename type>
	static real InnerProduct(type* x, type* y, int length)
	{
		real result = 0;
		for (int i = 0; i < length; ++i)
			result += x[i] * y[i];
		return result;
	}
	
	static real Sigmoid(const real f);
	static real d2Sigmoid(const real f);
	static void SoftMax(real* s, real* result, int size);
	static void MatVecProd(real* mat, real* vec, int m, int n, real* prod, bool is_trans);
	static void MatPlusMat(real* mat0, real* mat1, real c, int m, int n);
	static void CaseTransfer(char* word, int len);
	static bool IsValid(const real & f);
	static real TruncatedLog(const real& f);
	static real MaxValue(real* p_start, int len);
};