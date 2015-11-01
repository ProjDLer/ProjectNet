#include "util.h"

const char* Opt::train_file = NULL;
const char* Opt::save_vocab_file = NULL;
const char* Opt::read_vocab_file = NULL;
const char* Opt::binary_embedding_file = NULL;
const char* Opt::text_embedding_file = NULL;
const char* Opt::sw_file = NULL;
const char* Opt::out_relat_text = NULL;
const char* Opt::out_relat_binary = NULL;

bool Opt::hs = false;
int Opt::negative_num = 0;
int Opt::output_binary = 1;
real Opt::sample = 0;
int Opt::embeding_size = 0;
int Opt::thread_cnt = 1;
int Opt::window_size = 5;
int Opt::min_count = 5;
real Opt::init_learning_rate = (real)0.025;
int Opt::epoch = 1;
bool Opt::stopwords = false;
int Opt::batch_size = 1;
int Opt::know_iter = 1;
int Opt::head_relat_rank = 1;
int Opt::tail_relat_rank = 1;

real Opt::lambda = (real)1.0;
real Opt::margin = 1;
real Opt::know_update_per_progress = 0;
bool Opt::update_mat = 1;
bool Opt::use_know_in_text = false;
bool Opt::use_tail_mat = true;
bool Opt::is_diag = false;

const char* Opt::huff_tree_file = NULL;
const char* Opt::outputlayer_binary_file = NULL;
const char* Opt::outputlayer_text_file = NULL;

bool Opt::use_relation = true; //Indicate wheter the relation embedding is updated and used
bool Opt::act_relat = true;
bool Opt::act_relat_mat = false;
bool Opt::sig_relat = false;
real Opt::relat_neg_weight = 1;
const char* Opt::relation_file = NULL;

int* Sampler::table = NULL;
std::default_random_engine Sampler::generator;
std::uniform_int_distribution<int> Sampler::int_distribution(0, table_size - 1);

extern Dictionary* g_dict;
extern HuffmanEncoder g_encoder;

bool Util::ReadWord(char *word, FILE *fin)
{
	int idx = 0;
	char ch;
	while (!feof(fin))
	{
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
		{
			if (idx > 0)
			{
				if (ch == '\n')
					ungetc(ch, fin);
				break;
			}
			if (ch == '\n')
			{
				strcpy(word, (char *)"</s>");
				return true;
			}
			else continue;
		}
		word[idx++] = ch;
		if (idx >= MAX_STRING - 1) idx--;   // Truncate too long words
	}
	word[idx] = 0;
	return idx != 0;
}

void Util::SaveVocab()
{
	FILE* fid = fopen(Opt::save_vocab_file, "w");
	for (int i = 0; i < g_dict->Size(); ++i)
	{
		const WordInfo* info = g_dict->GetWordInfo(i);
		fprintf(fid, "%s %d\n", info->word.c_str(), info->freq);
	}
	fclose(fid);
}

bool Util::IsValid(const real& f)
{
	return f < 1 || f >= 1;
}

real Util::Sigmoid(const real f)
{
	if (f >= MAX_EXP)
		return 1;
	if (f <= -MAX_EXP)
		return 0;
	return 1 / (1 + exp(-f));
}

//the derivative of 2sigmoid(x) - 1
real Util::d2Sigmoid(const real f)
{
	return (1 - f) * (1 + f) / 2;
}

real Util::TruncatedLog(const real& f)
{
	return abs(f) > eps ? log(f) : MIN_LOG;
}

void Util::SoftMax(real* s, real* result, int size)
{
	real sum = 0, max_v = s[0];
	for (int j = 1; j < size; ++j)
		max_v = std::max(max_v, s[j]);
	for (int j = 0; j < size; ++j)
		sum += exp(s[j] - max_v);
	for (int j = 0; j < size; ++j)
		result[j] = exp(s[j] - max_v) / sum;
}

//prod = mat * vec, where mat\in R^{m*n}, vec \in R^{n*1}, if is_trans = true, then calculate prod = mat' * vec, where vec\in R^{m*1}
void Util::MatVecProd(real* mat, real* vec, int m, int n, real* prod, bool is_trans)
{
	memset(prod, 0, sizeof(real)* (is_trans? n : m));
	for (int i = 0; i < m; ++i)
		for (int j = 0; j < n; ++j)
		{
			if (abs(mat[i * n + j]) < eps)
				continue;
			if (!is_trans)
				prod[i] += mat[i * n + j] * vec[j];
			else
				prod[j] += mat[i * n + j] * vec[i];
		}
}

//mat0 += c * mat1
void Util::MatPlusMat(real* mat0, real* mat1, real c, int m, int n)
{
	for (int i = 0; i < m * n; ++i)
		mat0[i] += c * mat1[i];
}

void Util::CaseTransfer(char  *word, int len)
{
	for (int i = 0; i < len; i++)
	if (word[i] >= 'A' && word[i] <= 'Z') word[i] += 'a' - 'A';
}

real Util::MaxValue(real* p_start, int len)
{
	real max_v = p_start[0];
	for (int i = 1; i < len; ++i)
		max_v = std::max(max_v, p_start[i]);
	return max_v;
}
