#include "neuralnet.h"
#include "wordparams.h"
#include "HuffmanEncoder.h"
#include "reader.h"
#include <algorithm>
#include "relation.h"

real *NN::input_embedding = NULL;
real *NN::hs_embedding = NULL;
real *NN::negative_embedding = NULL;

real NN::learning_rate = 0; 
int NN::cur_epoch = 0;
int NN::embedding_cnt = 0;
real *NN::relat_prior = NULL;
real NN::progress = 0;

long long NN::last_train_count = 0;
long long NN::train_count = 0;
long long NN::train_count_total = 0;
long long NN::train_count_actual = 0;

extern Dictionary* g_dict;
extern HuffmanEncoder g_encoder;
extern Reader g_reader;
extern CRelation g_relat;
extern FILE* g_debug;

void NN::InitInputEmbedding()
{
	NN::input_embedding = (real*)malloc((long long)NN::embedding_cnt * Opt::embeding_size * sizeof(real));
	for (long long i = 0; i < embedding_cnt; i++) 
		for (int j = 0; j < Opt::embeding_size; j++)
			NN::input_embedding[i * Opt::embeding_size + j] = (rand() / (real)RAND_MAX - 0.5) / Opt::embeding_size;
}

void NN::InitNet()
{
	progress = 0;

	NN::InitInputEmbedding();
	NN::relat_prior = (real*)malloc(embedding_cnt * sizeof(real));
	if (Opt::hs) 
		NN::hs_embedding = (real*)calloc((long long)g_dict->Size() * Opt::embeding_size, sizeof(real));
	if (Opt::negative_num)
		NN::negative_embedding = (real*)calloc((long long)g_dict->Size() * Opt::embeding_size, sizeof(real));
}

void NN::SaveEmbedding()
{
	int str_len;
	char word[MAX_WORD_SIZE];
	if (Opt::output_binary)
	{
		FILE* fid = fopen(Opt::binary_embedding_file, "wb");
		int dic_size = g_dict->Size();
		fwrite(&dic_size, sizeof(int), 1, fid);
		fwrite(&Opt::embeding_size, sizeof(int), 1, fid);
		
		for (int i = 0; i < g_dict->Size(); ++i) 
		{
			strcpy(word, g_dict->GetWordInfo(i)->word.c_str());
			str_len = strlen(word);
			fwrite(&str_len, sizeof(int), 1, fid);
			fwrite(word, sizeof(char), str_len + 1, fid);
			fwrite(&NN::input_embedding[i * Opt::embeding_size], sizeof(real), Opt::embeding_size, fid);
		}
		fprintf(fid, "\n");
		fclose(fid);
    }
	if (Opt::output_binary % 2 == 0)
	{
		FILE* fid = fopen(Opt::text_embedding_file, "wb");
		fprintf(fid, "%d %d\n", g_dict->Size(), Opt::embeding_size);
		for (int i = 0; i < g_dict->Size(); ++i) 
		{
			fprintf(fid, "%s ", g_dict->GetWordInfo(i)->word.c_str());
			for (int j = 0; j < Opt::embeding_size; ++j) 
				fprintf(fid, "%lf ", NN::input_embedding[i * Opt::embeding_size + j]);
			fprintf(fid, "\n");
		}
		fprintf(fid, "\n");
		fclose(fid);
	}
}

void NN::FormerSaveEmbedding()
{
	if (Opt::output_binary)
	{
		FILE* fid = fopen("orig_emb.bin", "wb");
		fprintf(fid, "%d %d\n", g_dict->Size(), Opt::embeding_size);
		for (int i = 0; i < g_dict->Size(); ++i)
		{
			fprintf(fid, "%s ", g_dict->GetWordInfo(i)->word.c_str());
			for (int j = 0; j < Opt::embeding_size; ++j)
				fwrite(&NN::input_embedding[i * Opt::embeding_size + j], sizeof(real), 1, fid);
		}
		fprintf(fid, "\n");
		fclose(fid);
	}
}

void NN::SaveMultiInputEmbedding()
{
	if (Opt::output_binary)
	{
		FILE* fid = fopen(Opt::binary_embedding_file, "wb");
		fprintf(fid, "%d %d %d\n", g_dict->Size(), NN::embedding_cnt, Opt::embeding_size);
		for (int i = 0; i < g_dict->Size(); ++i)
		{
			fprintf(fid, "%s %d ", g_dict->GetWordInfo(i)->word.c_str(), WordParams::relat_cnt[i]);
			for (int j = 0; j < WordParams::relat_cnt[i]; ++j)
			{
				fwrite(WordParams::p_relat_prior[i] + j, sizeof(real), 1, fid);
				fwrite(WordParams::p_embedding[i] + j * Opt::embeding_size, sizeof(real), Opt::embeding_size, fid); 
			}
			fprintf(fid, "\n");
		}
		fclose(fid);
	}
	if (Opt::output_binary % 2 == 0)
	{
		FILE* fid = fopen(Opt::text_embedding_file, "w");
		fprintf(fid, "%d %d %d\n", g_dict->Size(), NN::embedding_cnt, Opt::embeding_size);
		for (int i = 0; i < g_dict->Size(); ++i)
		{
			fprintf(fid, "%s %d\n", g_dict->GetWordInfo(i)->word.c_str(), WordParams::relat_cnt[i]);
			for (int j = 0; j < WordParams::relat_cnt[i]; ++j)
			{
				fprintf(fid, "%.2f", WordParams::p_relat_prior[i][j]);
				for (int k = 0; k < Opt::embeding_size; ++k)
					fprintf(fid, " %.2f", WordParams::p_embedding[i][j * Opt::embeding_size + k]);
				fprintf(fid, "\n");
			}
			fprintf(fid, "\n");
		}
		fclose(fid);
	}
}

void NN::SaveHuffEncoder()
{
	FILE* fid = fopen(Opt::huff_tree_file, "w");
	fprintf(fid, "%d\n", g_dict->Size());
	for (int i = 0; i < g_dict->Size(); ++i)
	{
		fprintf(fid, "%s", g_dict->GetWordInfo(i)->word.c_str());
		auto info = g_encoder.GetLabelInfo(i);
		fprintf(fid, " %d", info->codelen);
		for (int j = 0; j < info->codelen; ++j)
			fprintf(fid, " %d", info->code[j]);
		for (int j = 0; j < info->codelen; ++j)
			fprintf(fid, " %d", info->point[j]);
		fprintf(fid, "\n");
	}
	fclose(fid);
}

void NN::SaveOutLayerEmbedding()
{
	if (Opt::output_binary)
	{
		FILE* fid = fopen(Opt::outputlayer_binary_file, "wb");
		fprintf(fid, "%d %d\n", g_dict->Size(), Opt::embeding_size);
		fwrite(NN::hs_embedding, sizeof(real), Opt::embeding_size * g_dict->Size(), fid); 
		fclose(fid);
	}
	if (Opt::output_binary % 2 == 0)
	{
		FILE* fid = fopen(Opt::outputlayer_text_file, "w");
		fprintf(fid, "%d %d\n", g_dict->Size(), Opt::embeding_size);
		for (int i = 0; i < g_dict->Size(); ++i)
		{
			real* output_embedding = NN::hs_embedding + i * Opt::embeding_size;
			for (int k = 0; k < Opt::embeding_size; ++k)
				fprintf(fid, "%.2f ", output_embedding[k]);
			fprintf(fid, "\n");
		}
		fclose(fid);
	}
}

void NN::UpdateInputEmbeddings(int word_input, int length, long long* points, int* labels, real* estimation, real* f_m, real* LTg_out, real* RTLTg_out, real* input_grads)
{
	real* output_embedding, *outputEmbeddingBase = Opt::hs ? NN::hs_embedding : NN::negative_embedding;

	int step = Opt::hs ? MAX_CODE_LENGTH : (Opt::negative_num + 1);

	for (int relat_idx = 0; relat_idx < WordParams::relat_cnt[word_input]; ++relat_idx)
	{
		long long fidx = relat_idx * step;
		
		for (int d = 0; d < length; ++d, ++fidx)
		{
			output_embedding = outputEmbeddingBase + points[d] * Opt::embeding_size;
			if (relat_idx == 0)
			{
				f_m[fidx] = estimation[relat_idx] * (labels[d] - f_m[fidx]);
				Util::MatPlusMat(input_grads, output_embedding, f_m[fidx], Opt::embeding_size, 1);
			}
		}
	}
}

void NN::UpdateOutputEmbeddings(int word_input, int length, long long* points, real* estimation, real* g_m, real* input_backup)
{
	real* output_embedding;
	real* inputEmbedding = input_backup, *outputEmbeddingBase = Opt::hs ? NN::hs_embedding : NN::negative_embedding;
	int step = Opt::hs ? MAX_CODE_LENGTH : (Opt::negative_num + 1);

	for (int relat_idx = 0; relat_idx < WordParams::relat_cnt[word_input]; ++relat_idx, inputEmbedding += Opt::embeding_size)
	{
		long long fidx = relat_idx * step;
		for (int d = 0; d < length; ++d, ++fidx)
		{
			output_embedding = outputEmbeddingBase + points[d] * Opt::embeding_size;
			
			for (int j = 0; j < Opt::embeding_size; ++j)
				output_embedding[j] += NN::learning_rate * g_m[fidx] * inputEmbedding[j];
		}
	}

}

void NN::BatchUpdateEmbeddings(int word_input, int* lengths, int* labels, long long* points, int batch_size, real* gamma, real* fTable, real* input_backup,
	real* LTg_out, real* RTLTg_out, real* input_grads, UpdateDirection direction)
{
	real* f_m = fTable;
	real* estimation = gamma;
	long long* p_points = points;
	int* p_label = labels, step = Opt::hs ? MAX_CODE_LENGTH : (Opt::negative_num + 1);

	memset(input_grads, 0, sizeof(real)* Opt::embeding_size);

	for (int i = 0; i < batch_size; ++i, estimation += MAX_RELAT_CNT, f_m += MAX_RELAT_CNT * step, p_label += step, p_points += step)
	{
		if (direction == UpdateDirection::UPDATE_INPUT)
			NN::UpdateInputEmbeddings(word_input, lengths[i], p_points, p_label, estimation, f_m, LTg_out, RTLTg_out, input_grads);
		else
			NN::UpdateOutputEmbeddings(word_input, lengths[i], p_points, estimation, f_m, input_backup);
	}

	if (direction == UpdateDirection::UPDATE_INPUT)
	{
		real* inputEmbedding = WordParams::p_embedding[word_input];
		for (int i = 0; i < Opt::embeding_size; ++i)
			inputEmbedding[i] += NN::learning_rate / WordParams::relat_cnt[word_input] * input_grads[i];
	}
}

//Return the Q value
real NN::Estimate_Gamma_m(int word_input, int length, long long* points, int* labels, real* log_posterior, real* estimation, real* relat_prior, real* f_m)
{
	real* inputEmbedding = WordParams::p_embedding[word_input], *outputEmbeddingBase = Opt::hs? NN::hs_embedding : NN::negative_embedding;
	real f;
	
	int step = Opt::hs ? MAX_CODE_LENGTH : (Opt::negative_num + 1);
	for (int relat_idx = 0; relat_idx < WordParams::relat_cnt[word_input]; ++relat_idx, inputEmbedding += Opt::embeding_size)
	{
		log_posterior[relat_idx] = Util::TruncatedLog(relat_prior[relat_idx]);
		
		long long fidx = relat_idx * step;
		for (int d = 0; d < length; ++d, ++fidx)
		{
			f = Util::InnerProduct(inputEmbedding, outputEmbeddingBase + points[d] * Opt::embeding_size, Opt::embeding_size);
			f = Util::Sigmoid(f);
			f_m[fidx] = f;
			if (!labels[d])
				f = 1 - f;
			log_posterior[relat_idx] += Util::TruncatedLog(f);
		}
	}
	if (!Opt::use_know_in_text || WordParams::relat_cnt[word_input] == 1)
	{
		estimation[0] = 1;
		return log_posterior[0];
	}

	Util::SoftMax(log_posterior, estimation, WordParams::relat_cnt[word_input]);

	return Util::InnerProduct(log_posterior, estimation, WordParams::relat_cnt[word_input]);
}

real NN::ComputeLikelihood(int word_input, int length, long long* points, int* labels, real* log_posterior, real* estimation, real* relat_prior, real* f_m)
{
	real* inputEmbedding = WordParams::p_embedding[word_input], *outputEmbeddingBase = Opt::hs ? NN::hs_embedding : NN::negative_embedding;
	real f;

	int step = Opt::hs ? MAX_CODE_LENGTH : (Opt::negative_num + 1);
	for (int relat_idx = 0; relat_idx < WordParams::relat_cnt[word_input]; ++relat_idx, inputEmbedding += Opt::embeding_size)
	{
		log_posterior[relat_idx] = Util::TruncatedLog(relat_prior[relat_idx]);

		long long fidx = relat_idx * step;
		for (int d = 0; d < length; ++d, ++fidx)
		{
			f = Util::InnerProduct(inputEmbedding, outputEmbeddingBase + points[d] * Opt::embeding_size, Opt::embeding_size);
			f = Util::Sigmoid(f);
			f_m[fidx] = f;
			if (!labels[d])
				f = 1 - f;
			log_posterior[relat_idx] += Util::TruncatedLog(f);
		}
	}
	if (!Opt::use_know_in_text || WordParams::relat_cnt[word_input] == 1)
	{
		estimation[0] = 1;
		return log_posterior[0];
	}

	return Util::InnerProduct(log_posterior, estimation, WordParams::relat_cnt[word_input]);
}

