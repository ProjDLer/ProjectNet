#include <thread>
#include <string>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <assert.h>
#include <unordered_set>

#include "Dictionary.h"
#include "HuffmanEncoder.h"
#include "util.h"
#include "neuralnet.h"
#include "reader.h"
#include "WordParams.h"
#include "relation.h"

Dictionary* g_dict = new Dictionary();
HuffmanEncoder g_encoder;
Reader g_reader;
clock_t start_time;
CRelation g_relat;
FILE* g_debug;

void StartEpoch()
{
	start_time = clock();
	NN::train_count = NN::last_train_count = NN::train_count_actual = 0;
}

inline bool UpdateLearningRate()
{
	bool need_update_know = false;
	long long per_update_word_cnt = NN::train_count_total * Opt::know_update_per_progress;

	if (NN::train_count - NN::last_train_count > 10000)
	{	
		int last = Opt::know_update_per_progress > eps ? NN::train_count_actual / per_update_word_cnt : -1;
		
		NN::train_count_actual += NN::train_count - NN::last_train_count;
		if (Opt::know_update_per_progress > eps && NN::train_count_actual / per_update_word_cnt > last)
			need_update_know = true;
		
		NN::last_train_count = NN::train_count;
		clock_t now = clock();
		NN::progress = NN::train_count_actual / (real)(NN::train_count_total + 1);
		real obs_mat_ele[MAX_RELAT_RANK] = { 0 };

		if (Opt::is_diag)
			for (auto x : g_relat.head_diag_mat_ele[2])
				obs_mat_ele[x.first] = x.second;

			printf("%cAlpha: %f  Progress: %.2f%%  Speed: %.2fk  Maxv: %.4f",
				13, NN::learning_rate, NN::progress * 100, NN::train_count_actual / ((real)(now - start_time + 1) / (real)CLOCKS_PER_SEC * 1000) / (2 * Opt::window_size),
				Util::MaxValue(Opt::is_diag ? obs_mat_ele : g_relat.p_head_left_mat[1], Opt::is_diag ? Opt::head_relat_rank : Opt::embeding_size * Opt::head_relat_rank));
			
		fflush(stdout);
		real actual = NN::train_count_actual + (real)NN::cur_epoch * (NN::train_count_total + 1);
		NN::learning_rate = Opt::init_learning_rate * (1 - actual / (real)(NN::train_count_total + 1) / Opt::epoch);
		if (NN::learning_rate < 0.000025) NN::learning_rate = 0.000025;
	}
	return need_update_know;
}

inline void Maximize_Pi(int word_input, int batch_size, real* relat_prior, real* gamma)
{
	if(WordParams::relat_cnt[word_input] == 1)
	{
		relat_prior[0] = 1;
		return;
	}

	real* estimation;
	for (int relat_idx = 0; relat_idx < WordParams::relat_cnt[word_input]; ++relat_idx)
	{
		estimation = gamma;
		real new_alpha = 0;
		for (int j = 0; j < batch_size; ++j, estimation += MAX_RELAT_CNT)
			new_alpha += estimation[relat_idx];
	}
}

void TrainBatch(int word_input, std::vector<int>& outputs, int batch_size, int* lengths, int* labels, long long* points, real* gamma, real* gTable, real* input_backup,
	real* g_out, real* RInput, real* LTg_out, real* RTLTg_out, real* input_grads)
{
	NN::train_count += batch_size;

	real posterior[MAX_RELAT_CNT];
	real* relat_prior = WordParams::p_relat_prior[word_input];
	real *estimation, *f_m;
	real log_likelihood;
	int *p_labels, step = Opt::hs? MAX_CODE_LENGTH: (Opt::negative_num + 1);
	long long* p_points;
	int iter = 0;
	// compute and backup input embeddings
	
	memcpy(input_backup, WordParams::p_embedding[word_input], Opt::embeding_size * WordParams::relat_cnt[word_input] * sizeof(real));
	log_likelihood = 0;

	estimation = gamma;
	f_m = gTable;
	p_labels = labels;
	p_points = points;

	// E-Step
	for (int i = 0; i < batch_size; ++i, estimation += MAX_RELAT_CNT, f_m += MAX_RELAT_CNT * step, p_labels += step, p_points += step)
	{
		if (Opt::hs && iter == 0)
		{
			auto info = g_encoder.GetLabelInfo(outputs[i]);
			lengths[i] = info->codelen;
			for (int d = 0; d < info->codelen; ++d)
			{
				p_points[d] = info->point[d];
				p_labels[d] = info->code[d];
			}
		}
		else if (Opt::negative_num && iter == 0)
		{
			lengths[i] = Opt::negative_num + 1;
				
			p_points[0] = outputs[i];
			p_labels[0] = 1;
			for (int j = 0; j < Opt::negative_num; ++j)
			{
				int sampled;
				do
				{
					sampled = Sampler::NegativeSampling();
				} while (sampled == outputs[i]);
				p_points[j + 1] = sampled;
				p_labels[j + 1] = 0;
			}
		}
			
		log_likelihood += NN::Estimate_Gamma_m(word_input, lengths[i], p_points, p_labels, posterior, estimation, relat_prior, f_m);
	}
	// M-Step
	Maximize_Pi(word_input, batch_size, relat_prior, gamma);

	NN::BatchUpdateEmbeddings(word_input, lengths, labels, points, batch_size, gamma, gTable, input_backup, LTg_out, RTLTg_out, input_grads, UpdateDirection::UPDATE_INPUT);

	NN::BatchUpdateEmbeddings(word_input, lengths, labels, points, batch_size, gamma, gTable, input_backup, LTg_out, RTLTg_out, input_grads, UpdateDirection::UPDATE_OUTPUT);
		
	if (Opt::use_relation && Opt::lambda > eps && Opt::know_update_per_progress < eps)
	{
		auto it = g_relat.wordid2entityid_table.find(word_input);
		
		if (it != g_relat.wordid2entityid_table.end()) //update relational embedding 
		{
			g_relat.TrainRelation(word_input);
		}
	}

	if (UpdateLearningRate() && Opt::lambda > eps && Opt::know_update_per_progress > eps)
		g_relat.SeqTrainRelation();
}


void TrainModelThread()
{
	int sentence[MAX_SENTENCE_LENGTH + 2], sentence_length;

	std::vector<int>* outputs = new std::vector<int>[g_dict->Size()];
	for (int i = 0; i < g_dict->Size(); ++i)
		outputs[i].clear();
	int* sample_num = (int*)calloc(g_dict->Size(), sizeof(int));
	int word_output, word_input, cnt;

	//Init the tmp gradient vector for text
	int* labels = (int*)calloc(MAX_BATCH_SIZE * MAX_CODE_LENGTH, sizeof(int));
	long long* points = (long long*)calloc(MAX_BATCH_SIZE * MAX_CODE_LENGTH, sizeof(long long));
	int* lengths = (int*)calloc(MAX_BATCH_SIZE, sizeof(int));

	real *gamma = (real*)calloc((Opt::batch_size + Opt::window_size) * MAX_RELAT_CNT, sizeof(real));
	real *fTable = (real*)calloc((Opt::batch_size + Opt::window_size) * MAX_CODE_LENGTH * MAX_RELAT_CNT, sizeof(real));
	real *gTable = (real*)calloc((Opt::batch_size + Opt::window_size) * MAX_CODE_LENGTH * MAX_RELAT_CNT, sizeof(real));
	real *input_backup = (real*)calloc(Opt::embeding_size * MAX_RELAT_CNT, sizeof(real));
	real* g_out = (real*)calloc(Opt::embeding_size + Opt::window_size,  sizeof(real));
	real* RInput = (real*)calloc(MAX_RELAT_CNT * MAX_RELAT_RANK, sizeof(real));
	real *LTg_out = (real*)calloc(Opt::batch_size * MAX_RELAT_RANK * MAX_CODE_LENGTH * MAX_RELAT_CNT, sizeof(real));
	real* RTLTg_out = (real*)calloc(Opt::embeding_size + Opt::window_size, sizeof(real));
	real* input_grads = (real*)calloc(Opt::embeding_size + Opt::window_size, sizeof(real));
	
	//Init the tmp gradients vector for knowledge

	while (1)
	{
		sentence_length = g_reader.GetSentence(sentence);
		if (sentence_length == 0)
			break;
		for (int sentence_position = 0; sentence_position < sentence_length; ++sentence_position)
		{
			while (g_relat.know_lock);
			word_output = sentence[sentence_position];
			if (word_output == -1) continue;
			for (int i = 0; i < Opt::window_size * 2 + 1; ++i)
				if (i != Opt::window_size)
				{
					word_input = sentence_position - Opt::window_size + i;
					if (word_input < 0 || word_input >= sentence_length)
						continue;
					word_input = sentence[word_input];
					cnt = sample_num[word_input]++;
					if (outputs[word_input].size() >= Opt::batch_size)
						outputs[word_input][cnt] = word_output;
					else 
						outputs[word_input].push_back(word_output);
					cnt++;
					if (cnt >= Opt::batch_size)
					{
						TrainBatch(word_input, outputs[word_input], cnt, lengths, labels, points, gamma, fTable, input_backup,
							g_out, RInput, LTg_out, RTLTg_out, input_grads);
						sample_num[word_input] = 0;						
					}
				}
		}
	}
	for (int i = 0; i < g_dict->Size(); ++i)
		if (sample_num[i])
		{		
			TrainBatch(i, outputs[i], sample_num[i], lengths, labels, points, gamma, fTable, input_backup,
				g_out, RInput, LTg_out, RTLTg_out, input_grads);
			sample_num[word_input] = 0;
		}

	free(gamma);
	free(fTable);
	free(gTable);
	free(input_backup);
	free(g_out);
	free(RInput);
	free(LTg_out);
	free(RTLTg_out);
	free(input_grads);
}

void Train()
{
	if (Opt::negative_num)
		Sampler::SetNegativeSamplingDistribution(g_dict);
	NN::learning_rate = Opt::init_learning_rate;
	for (NN::cur_epoch = 0; NN::cur_epoch < Opt::epoch; ++NN::cur_epoch)
	{
		printf("\nEpoch %d/%d Starts\n", NN::cur_epoch + 1, Opt::epoch);
		StartEpoch();

		g_reader.Open(Opt::train_file, g_dict);
		std::vector<std::thread> thread_pool;
		for (int i = 0; i < Opt::thread_cnt; ++i)
		{
			thread_pool.push_back (
				std::thread([=]()
				{
					TrainModelThread();
				}
			));
		}
		for (int i = 0; i < Opt::thread_cnt; ++i)
		  thread_pool[i].join();
		g_reader.Close();
		Opt::init_learning_rate = NN::learning_rate;
	}
	printf("\nTraining Finished\n");
}

void LoadRelation()
{
	printf("begin loadding relations\n");
	g_relat.LoadListFromFile(Opt::relation_file);
	g_relat.BuildTableFromList();
	g_relat.InitKnowledgeEmb();
}

/*CRelation test_relat;
struct TestSample
{
	int idxes[4];
};


void ReadTestRelat()
{
	FILE* fid = fopen("D:\\Work\\DeepLearning\\KnowledgePowerdWordEmbedding\\dataset\\NeuralTensorNetwork\\data\\Freebase\\ana_test_files\\questions-fb13.txt", "r");
	char buf[2 * MAX_SENTENCE_LENGTH];
	fscanf(fid, "%s", buf);
	bool is_eof = false;
	int invalid_cnt = 0;
	std::string r_name;
	int r_idx, list_id = 0;
	while (!is_eof)
	{
		fscanf(fid, "%s", buf);
		r_name = std::string(buf);
		r_idx = g_relat.relation_name2id_table[r_name];
		int total = 0, seen = 0;
		while (true)
		{
			if (fscanf(fid, "%s", buf) == EOF)
			{
				is_eof = true;
				break;
			}
			if (buf[0] == ':')
				break;

			bool valid = true;
			TestSample sample;
			sample.idxes[0] = g_dict->GetWordIdx(buf);
			valid &= sample.idxes[0] != -1;
			for (int i = 1; i < 4; ++i)
			{
				fscanf(fid, "%s", buf);
				sample.idxes[i] = g_dict->GetWordIdx(buf);
				valid &= sample.idxes[i] != -1;
			}
			if (valid)
			{
				seen++;
				test_relat.list.push_back(StruRelation(sample.idxes[0], sample.idxes[1], r_idx));
			}
			else
				invalid_cnt++;
			total++;
		}
	}
	//for (int i = 0; i < 10; ++i)
		//printf("%s %s %s\n", g_dict->GetWordInfo(test_relat.list[i].w1)->word.c_str(), g_relat.relation_id2name_table[test_relat.list[i].r].c_str(), g_dict->GetWordInfo(test_relat.list[i].w2)->word.c_str());
}*/

int main(int argc, char* argv[])
{
	g_debug = fopen("debug.txt", "w");
	srand(time(NULL));
	Opt::ParseArgs(argc, argv);
	Reader::LoadVocab();

	NN::train_count_total = g_reader.total_words * 2 * Opt::window_size;
	if (Opt::save_vocab_file)
		Util::SaveVocab();
	if (Opt::use_relation)
		LoadRelation();

	WordParams::InitRelatCnt();
	NN::InitNet();
	WordParams::InitWordParams();

//#pragma omp parallel for
	
	Train();
	/*ReadTestRelat();

	NN::learning_rate = Opt::init_learning_rate;

	NN::learning_rate = Opt::init_learning_rate;

	for (int i = 0; i < Opt::know_iter; ++i)
	{
		//NN::learning_rate = Opt::init_learning_rate * (1 - (i + 0.0) / Opt::know_iter);
		real loss = g_relat.SeqTrainRelation();
	}

	//int acc_cnt = 0;
	acc_cnt = 0;
	printf("%d\n", g_relat.relat_size);
	for (int i = 0; i < g_relat.list.size(); ++i)
	{
		int best_r = -1; real best_score;
		for (int r = 0; r < g_relat.relat_size; ++r)
		{
			real r_score = g_relat.ComputeLoss(g_relat.list[i].w1, g_relat.list[i].w2, r);
			if (best_r == -1 || best_score > r_score)
			{
				best_r = r;
				best_score = r_score;
			}
		}
		if (rand() == 1)
		printf("%d,%d\n", i, best_r);
		if (best_r == g_relat.list[i].r)
			acc_cnt++;
	}
	printf("\nTraining total %d triples, right cnt: %d, ratio: %.4f\n", g_relat.list.size(), acc_cnt, acc_cnt * 1.0 / g_relat.list.size());
	
	acc_cnt = 0;
	int total = 0;

	for (int i = 0; i < test_relat.list.size(); ++i)
	{
		if ((rand() + 0.0) / RAND_MAX >= 0.005)
			continue;
		total++;
		int best_w = -1; real best_score;
		for (int w = 0; w < g_dict->Size(); ++w)
		{
			real w_score = g_relat.ComputeLoss(test_relat.list[i].w1, w, test_relat.list[i].r);
			if (best_w == -1 || best_score > w_score)
			{
				best_w = w;
				best_score = w_score;
			}
		}
		printf("%c%d", 13, i);
		if (best_w == test_relat.list[i].w2)
			acc_cnt++;
	}
	printf("\nTesting total %d triples, right cnt: %d, ratio: %.4f\n", total, acc_cnt, acc_cnt * 1.0 / total);*/

	/*NN::SaveMultiInputEmbedding();
	if (Opt::huff_tree_file)
		NN::SaveHuffEncoder();
	if (Opt::outputlayer_binary_file || Opt::outputlayer_text_file)
		NN::SaveOutLayerEmbedding();*/
	NN::SaveEmbedding();
	//NN::FormerSaveEmbedding();
	g_relat.SaveRelationEmb();

	return 0;
}
