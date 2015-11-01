#pragma once

#include "util.h"
#include <map>

struct NN
{
	static real *input_embedding;
	static real *hs_embedding;
	static real *negative_embedding;
	static real learning_rate;
	
	static real *relat_prior;
	static int cur_epoch;
	static int embedding_cnt;
	static real progress;
	static long long train_count_total, train_count_actual, last_train_count, train_count;

	static void InitInputEmbedding();
	static void InitNet();
	static void SaveEmbedding();
	static void FormerSaveEmbedding();

	static void SaveMultiInputEmbedding();

	static void SaveHuffEncoder();
	static void SaveOutLayerEmbedding();

	static void UpdateInputEmbeddings(int word_input, int length, long long* points, int* labels, real* estimation, real* f_m, real* g_out, real* RTLTout, real* input_grads);
	static void UpdateOutputEmbeddings(int word_input, int length, long long* points, real* estimation, real* f_m, real* input_backup);

	static void UpdateRelationMatrix(int word_input, int length, long long* points, int* labels, real* estimation, real* f_m, real* g_out, real* LTout, real* RInput, real* Lgrads, real* Rgrads);

	static void BatchUpdateEmbeddings(int word_input, int* lengths, int* labels, long long* points, int batch_size, real* gamma, real* fTable, real* input_backup, real* LTg_out, real* RTLTg_out, real* input_grads, UpdateDirection direction);
	
	static real Estimate_Gamma_m(int word_input, int length, long long* points, int* labels, real* log_posterior, real* estimation, real* relat_prior, real* f_m);
	static real ComputeLikelihood(int word_input, int length, long long* points, int* labels, real* log_posterior, real* estimation, real* relat_prior, real* f_m);
};