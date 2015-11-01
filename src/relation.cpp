#pragma once
#include "relation.h"
#include "neuralnet.h"
#include "wordparams.h"
#include "reader.h"
#include <thread>
#include <iostream>

extern Dictionary* g_dict;
extern FILE* g_debug;
extern Reader g_reader;

bool cmp_by_first_entity(StruRelation x, StruRelation y)
{
	if (x.w1 < y.w1) return true;
	else if (x.w1 == y.w1){
		if (x.w2 < y.w2) return true;
		else if (x.w2 == y.w2) {
			if (x.r < y.r) return true;
		}
	}
	return false;
}

bool cmp_by_second_entity(StruRelation x, StruRelation y)
{
	if (x.w2 < y.w2) return true;
	else if (x.w2 == y.w2){
		if (x.w1 < y.w1) return true;
		else if (x.w1 == y.w1) {
			if (x.r < y.r) return true;
		}
	}
	return false;
}

bool myequal(StruRelation x, StruRelation y)
{
	if (x.w1 == y.w1 && x.w2 == y.w2 && x.r == y.r) return true;
	return false;
}

CRelation::CRelation()
{
	relat_size = 0;
	generator.seed(time(NULL));
}

CRelation::~CRelation()
{
}

void CRelation::LoadListFromFile(const char * fname)
{
	if (!Opt::relation_file || !Opt::use_relation)	return;
	FILE* fid = fopen(fname, "r");
	char w1[200], w2[200], r[200];
	int w1_idx, w2_idx, relat_idx;
	while (fscanf(fid, "%s %s %s", w1, w2, r) != EOF)
	{
		Util::CaseTransfer(w1, strlen(w1));
		Util::CaseTransfer(w2, strlen(w2));
		w1_idx = g_dict->GetWordIdx(w1);
		w2_idx = g_dict->GetWordIdx(w2);

		if (w1_idx == -1 || w2_idx == -1)
			continue;

		int entity_id = wordid2entityid_table.size();
		if (wordid2entityid_table.find(w1_idx) == wordid2entityid_table.end())
		{
			wordid2entityid_table.insert(std::make_pair(w1_idx, entity_id));
			entityid2wordid_table[entity_id] = w1_idx;
		}

		entity_id = wordid2entityid_table.size();
		if (wordid2entityid_table.find(w2_idx) == wordid2entityid_table.end())
		{
			wordid2entityid_table.insert(std::make_pair(w2_idx, entity_id));
			entityid2wordid_table[entity_id] = w2_idx;
		}

		entity_id = wordid2entityid_table.size();
		if (entity_id > entity_size)
			entity_size = entity_id;

		if (relation_name2id_table.find(std::string(r)) == relation_name2id_table.end())
		{
			relat_idx = relation_name2id_table.size();
			relation_name2id_table[std::string(r)] = relat_idx;
			relation_id2name_table[relat_idx] = std::string(r);
		}
		else
			relat_idx = relation_name2id_table[std::string(r)];

		if (relat_idx + 1 > relat_size) relat_size = relat_idx + 1;
		bool input_valid;
		input_valid = (((w1_idx >= 0) && (w1_idx < g_dict->Size())) && ((w2_idx >= 0) && (w2_idx < g_dict->Size())));
		if (input_valid) {
			list.push_back(StruRelation(w1_idx, w2_idx, relat_idx));
			list_reverse.push_back(StruRelation(w1_idx, w2_idx, relat_idx));
			list_in_know_space.push_back(StruRelation(wordid2entityid_table[w1_idx], wordid2entityid_table[w2_idx], relat_idx));
			list_reverse_in_know_space.push_back(StruRelation(wordid2entityid_table[w1_idx], wordid2entityid_table[w2_idx], relat_idx));
		}
	}
	sample_num = 0;

	FILE* statistics_file = fopen("stat.txt", "w");
	real hr_ratio[MAX_RELAT_CNT], rt_ratio[MAX_RELAT_CNT];
	for (int r = 0; r < relat_size; ++r)
	{
		hr_ratio[r] = 0;
		rt_ratio[r] = 0;
	}

	relat_triple_size = (int*)calloc(relat_size, sizeof(int));

	std::map<int, int> h_cnt[MAX_RELAT_CNT], t_cnt[MAX_RELAT_CNT];
	for (int i = 0; i < list.size(); ++i)
	{
		int h = list[i].w1, r = list[i].r, t = list[i].w2;
		if (h_cnt[r].find(h) == h_cnt[r].end())
			h_cnt[r][h] = 1;
		else
			h_cnt[r][h]++;

		if (t_cnt[r].find(t) == t_cnt[r].end())
			t_cnt[r][t] = 1;
		else
			t_cnt[r][t]++;
		relat_triple_size[list[i].r]++;
	}

	for (int r = 0; r < relat_size; ++r)
	{
		for (auto&x : h_cnt[r])
			hr_ratio[r] += x.second;
		for (auto& x : t_cnt[r])
			rt_ratio[r] += x.second;

		fprintf(statistics_file, "%s, h_ratio: %.4f, t_ratio: %.4f\n", relation_id2name_table[r].c_str(), hr_ratio[r] * 1.0 / h_cnt[r].size(), rt_ratio[r] * 1.0 / t_cnt[r].size());
	}
	fclose(statistics_file);
	
	printf("Relation tuple number: %I64d, relation type number: %I64d, entities number: %I64d\n", list.size(), relat_size, entity_size);
}

void CRelation::BuildTableFromList()
{
	//Build head_idx_range_table
	std::sort(list.begin(), list.end(), cmp_by_first_entity);
	auto it = unique(list.begin(), list.end(), myequal);
	list.resize(it - list.begin());
	for (int i = 0; i < list.size();) {
		int j = i;
		while (j < list.size() && list[j].w1 == list[i].w1) j++;
		head_idx_range_table[list[i].w1] = std::pair<int, int>(i, j);
		i = j;
	}

	/*std::map<std::pair<int, int>, int> hr_pair_map;
	for (auto triple : list)
	{
		std::pair<int, int> hr = std::pair<int, int>(triple.w1, triple.r);
		hr_pair_map[hr]++;
	}
	long long hr_pair_cnt = hr_pair_map.size(), dup_hr_pair_cnt = 0;
	for (auto hr_pair_info : hr_pair_map)
		if (hr_pair_info.second > 1)
			dup_hr_pair_cnt++;
	printf("total hr pairs: %I64d, dup hr pairs: %I64d, ratio %.4f\n", hr_pair_cnt, dup_hr_pair_cnt, (0.0 + dup_hr_pair_cnt) / hr_pair_cnt);
	*/
	//Build tail_idx_range_table
	std::sort(list_reverse.begin(), list_reverse.end(), cmp_by_second_entity);
	it = unique(list_reverse.begin(), list_reverse.end(), myequal);
	list_reverse.resize(it - list_reverse.begin());
	for (int i = 0; i < list_reverse.size();) {
		int j = i;
		while (j < list_reverse.size() && list_reverse[j].w2 == list_reverse[i].w2) j++;
		tail_idx_range_table[list_reverse[i].w2] = std::pair<int, int>(i, j);
		i = j;
	}
}

int CRelation::SampleWordIdx()
{
	return distribution(generator);
}

int CRelation::SampleRelatIdx()
{
	return r_distribution(generator);
}

bool CRelation::ContainsEntity(int word_id) { return wordid2entityid_table.find(word_id) != wordid2entityid_table.end(); }

void CRelation::InitKnowledgeEmb()
{
	update_know_cnt = 0;

	if (!Opt::is_diag)
	{
		relat_head_left_matrix = (real*)calloc(relat_size * Opt::head_relat_rank * Opt::embeding_size, sizeof(real));
		relat_head_right_matrix = (real*)calloc(relat_size * Opt::head_relat_rank * Opt::embeding_size, sizeof(real));
		relat_tail_left_matrix = (real*)calloc(relat_size * Opt::tail_relat_rank * Opt::embeding_size, sizeof(real));
		relat_tail_right_matrix = (real*)calloc(relat_size * Opt::tail_relat_rank * Opt::embeding_size, sizeof(real));
	}

	if (!Opt::is_diag && Opt::act_relat_mat)
	{
		relat_actual_right_matrix = (real*)calloc(relat_size * Opt::tail_relat_rank * Opt::embeding_size, sizeof(real));
		relat_actual_left_matrix = (real*)calloc(relat_size * Opt::head_relat_rank * Opt::embeding_size, sizeof(real));
		p_actual_left_mat = (real**)malloc(relat_size * sizeof(real*));
		p_actual_right_mat = (real**)malloc(relat_size * sizeof(real*));
	}

	relat_emb_vecs = (real*)calloc(relat_size * Opt::embeding_size, sizeof(real));
	if (Opt::act_relat)
		relat_actual_emb_vecs = (real*)calloc(relat_size * Opt::embeding_size, sizeof(real));
	
	if (!Opt::is_diag)
	{
		p_head_right_mat = (real**)malloc(relat_size * sizeof(real*));
		p_head_left_mat = (real**)malloc(relat_size * sizeof(real*));
		p_tail_right_mat = (real**)malloc(relat_size * sizeof(real*));
		p_tail_left_mat = (real**)malloc(relat_size * sizeof(real*));
	}

	p_relat_emb = (real**)malloc(relat_size * sizeof(real*));
	if (Opt::act_relat)
		p_relat_act_emb = (real**)malloc(relat_size * sizeof(real*));
	
	std::uniform_int_distribution<int> rank_distribution = std::uniform_int_distribution<int>(0, Opt::embeding_size - 1);

	for (int i = 0; i < relat_size; i++) 
	{
		if (!Opt::is_diag)
		{
			p_head_right_mat[i] = relat_head_right_matrix + i * Opt::head_relat_rank * Opt::embeding_size;
			p_head_left_mat[i] = relat_head_left_matrix + i * Opt::head_relat_rank * Opt::embeding_size;
			p_tail_right_mat[i] = relat_tail_right_matrix + i * Opt::tail_relat_rank * Opt::embeding_size;
			for (int j = 0; j < std::min(Opt::head_relat_rank, Opt::embeding_size); ++j)
				p_head_right_mat[i][j * Opt::embeding_size + j] = 1;
				
			for (int j = 0; j < std::min(Opt::tail_relat_rank, Opt::embeding_size); ++j)
				p_tail_right_mat[i][j * Opt::embeding_size + j] = 1;

			p_tail_left_mat[i] = relat_tail_left_matrix + i * Opt::tail_relat_rank * Opt::embeding_size;
			for (int j = 0; j < std::min(Opt::head_relat_rank, Opt::embeding_size); ++j)
				p_head_left_mat[i][j * Opt::head_relat_rank + j] = 1;
				
			for (int j = 0; j < std::min(Opt::tail_relat_rank, Opt::embeding_size); ++j)
				p_tail_left_mat[i][j * Opt::tail_relat_rank + j] = 1;

			if (Opt::act_relat_mat)
			{
				p_actual_left_mat[i] = relat_actual_left_matrix + i * Opt::embeding_size * Opt::head_relat_rank;
				p_actual_right_mat[i] = relat_actual_right_matrix + i * Opt::embeding_size * Opt::tail_relat_rank;
			}
		}
		else
		{
			int idx;
			for (int j = 0; j < Opt::head_relat_rank; ++j)
			{
				do
				{
					idx = rank_distribution(generator);
				} while (head_diag_mat_ele[i].find(idx) != head_diag_mat_ele[i].end());
				head_diag_mat_ele[i].insert(std::make_pair(idx, 1.0));
			}

			for (int j = 0; j < Opt::tail_relat_rank; ++j)
			{
				do
				{
					idx = rank_distribution(generator);
				} while (tail_diag_mat_ele[i].find(idx) != tail_diag_mat_ele[i].end());
				tail_diag_mat_ele[i].insert(std::make_pair(idx, 1.0));
			}
		}

		//Init the relation embedding vectors
		p_relat_emb[i] = relat_emb_vecs + i * Opt::embeding_size;
		for (int j = 0; j < Opt::embeding_size; ++j)
			p_relat_emb[i][j] = (rand() / (real)RAND_MAX - 0.5) / Opt::embeding_size;

		if (Opt::act_relat)
			p_relat_act_emb[i] = relat_actual_emb_vecs + i * Opt::embeding_size;
			
		ConstrainParameters(i);
	}
	distribution = std::uniform_int_distribution<int>(0, g_dict->Size() - 1);
	r_distribution = std::uniform_int_distribution<int>(0, relat_size - 1);
	printf("Init relation completed\n");
}

void CRelation::SaveRelationEmb()
{
	FILE* text_fid, *bin_fid;
	real* embedding;
	if (Opt::out_relat_text != NULL)
		text_fid = fopen(Opt::out_relat_text, "w"); //Record matrix/relation embedding/entity embedding in text
	if (Opt::out_relat_binary != NULL)
		bin_fid = fopen(Opt::out_relat_binary, "wb"); //Record matrix/relation embedding in binary

	if (Opt::out_relat_text != NULL)
	{
		fprintf(text_fid, "%d %d %d %d %d\n", Opt::embeding_size, Opt::head_relat_rank, Opt::tail_relat_rank, relat_size, Opt::is_diag);
		fprintf(text_fid, "The relation embedding:\n");

		for (int i = 0; i < relat_size; ++i) //Record the relation embedding vectors
		{
			embedding = p_relat_emb[i];
			fprintf(text_fid, "%s ", relation_id2name_table[i].c_str());
			for (int j = 0; j < Opt::embeding_size; ++j)
				fprintf(text_fid, "%.4f ", embedding[j]);
			fprintf(text_fid, "\n");
			
			if (!Opt::is_diag)
			{
				for (int j = 0; j < Opt::embeding_size; ++j)
				{
					for (int k = 0; k < Opt::head_relat_rank; ++k)
						fprintf(text_fid, "%.4f ", p_head_left_mat[i][j * Opt::head_relat_rank + k]);
					fprintf(text_fid, "\n");
				}

				for (int j = 0; j < Opt::head_relat_rank; ++j)
				{
					for (int k = 0; k < Opt::embeding_size; ++k)
						fprintf(text_fid, "%.4f ", p_head_right_mat[i][j * Opt::embeding_size + k]);
					fprintf(text_fid, "\n");
				}
				for (int j = 0; j < Opt::embeding_size; ++j)
				{
					for (int k = 0; k < Opt::tail_relat_rank; ++k)
						fprintf(text_fid, "%.4f ", p_tail_left_mat[i][j * Opt::tail_relat_rank + k]);
					fprintf(text_fid, "\n");
				}

				for (int j = 0; j < Opt::tail_relat_rank; ++j)
				{
					for (int k = 0; k < Opt::embeding_size; ++k)
						fprintf(text_fid, "%.4f ", p_tail_right_mat[i][j * Opt::embeding_size + k]);
					fprintf(text_fid, "\n");
				}
				
			}
			else
			{
				for (int j = 0; j < Opt::embeding_size; ++j)
				{
					for (int k = 0; k < Opt::embeding_size; ++k)
					{
						if (j != k || head_diag_mat_ele[i].find(k) == head_diag_mat_ele[i].end())
							fprintf(text_fid, "0.0000 ");
						else
							fprintf(text_fid, "%.4f ", head_diag_mat_ele[i][k]);
					}
					fprintf(text_fid, "\n");
				}

				for (int j = 0; j < Opt::embeding_size; ++j)
				{
					for (int k = 0; k < Opt::embeding_size; ++k)
					{
						if (j != k || tail_diag_mat_ele[i].find(k) == tail_diag_mat_ele[i].end())
							fprintf(text_fid, "0.0000 ");
						else
							fprintf(text_fid, "%.4f ", tail_diag_mat_ele[i][k]);
					}
					fprintf(text_fid, "\n");
				}
			}
			fprintf(text_fid, "\n");
		}
	}

	if (Opt::out_relat_binary != NULL)
	{
		fwrite(&Opt::embeding_size, sizeof(int), 1, bin_fid);
		fwrite(&Opt::head_relat_rank, sizeof(int), 1, bin_fid);
		fwrite(&Opt::tail_relat_rank, sizeof(int), 1, bin_fid);
		fwrite(&relat_size, sizeof(int), 1, bin_fid);
		fwrite(&Opt::use_tail_mat, sizeof(bool), 1, bin_fid);
		fwrite(&Opt::is_diag, sizeof(bool), 1, bin_fid);

		char p[MAX_WORD_SIZE];
		int str_len;
		for (int i = 0; i < relat_size; ++i)
		{
			strcpy(p, relation_id2name_table[i].c_str());
			str_len = strlen(p);
			fwrite(&str_len, sizeof(int), 1, bin_fid);
			fwrite(p, sizeof(char), str_len + 1, bin_fid);
			fwrite(p_relat_emb[i], sizeof(real), Opt::embeding_size, bin_fid);
			if (!Opt::is_diag)
			{
				fwrite(p_head_left_mat[i], sizeof(real), Opt::head_relat_rank * Opt::embeding_size, bin_fid);
				fwrite(p_head_right_mat[i], sizeof(real), Opt::embeding_size * Opt::head_relat_rank, bin_fid);
				fwrite(p_tail_left_mat[i], sizeof(real), Opt::tail_relat_rank * Opt::embeding_size, bin_fid);
				fwrite(p_tail_right_mat[i], sizeof(real), Opt::embeding_size * Opt::tail_relat_rank, bin_fid);
			}
			else
			{
				for (auto x : head_diag_mat_ele[i])
				{
					fwrite(&(x.first), sizeof(int), 1, bin_fid);
					fwrite(&(x.second), sizeof(real), 1, bin_fid);
				}
				for (auto x : tail_diag_mat_ele[i])
				{
					fwrite(&(x.first), sizeof(int), 1, bin_fid);
					fwrite(&(x.second), sizeof(real), 1, bin_fid);
				}
			}
		}
	}

	printf("Relation embedding saved!\n");
	if (Opt::out_relat_text != NULL)
		fclose(text_fid);
	if (Opt::out_relat_binary != NULL)
		fclose(bin_fid);
}

void CRelation::TrainRelation(int centerWord)
{
	int w1, r, w2;
	auto it = head_idx_range_table.find(centerWord);
	w1 = centerWord;
	
	if (it != head_idx_range_table.end()) //For all the tuples in which centerword is head
	{
		for (int i = it->second.first; i < it->second.second; ++i)
		{
			r = list[i].r;
			w2 = list[i].w2;
			TrainRelatTriple(w1, true, r, w2);
		}
	}
	
	it = tail_idx_range_table.find(centerWord);
	w2 = centerWord;
	if (it != tail_idx_range_table.end()) //For all the tuples in which centerword is tail
	{
		for (int i = it->second.first; i < it->second.second; ++i)
		{
			w1 = list_reverse[i].w1;
			r = list_reverse[i].r;
			TrainRelatTriple(w1, false, r, w2);
		}
	}
}

real CRelation::ComputeLoss(int head, int tail, int r)
{
	real Qh_h[MAX_RELAT_RANK], Qt_t[MAX_RELAT_RANK], off_vec[MAX_EMBEDDING_SIZE];
	real PtQt_t[MAX_EMBEDDING_SIZE];
	
	real* head_embedding = WordParams::p_embedding[head];
	real* tail_embedding = WordParams::p_embedding[tail];
	real* relat_embedding = Opt::act_relat ? p_relat_act_emb[r] : p_relat_emb[r];

	if (Opt::use_tail_mat)
	{
		if (!Opt::is_diag)
		{
			Util::MatVecProd(p_tail_right_mat[r], tail_embedding, Opt::tail_relat_rank, Opt::embeding_size, Qt_t, false);
			Util::MatVecProd(p_tail_left_mat[r], Qt_t, Opt::embeding_size, Opt::tail_relat_rank, PtQt_t, false);
		}
		else
		{
			memset(PtQt_t, 0, sizeof(real)* Opt::embeding_size);
			for (auto x : tail_diag_mat_ele[r])
				PtQt_t[x.first] = x.second * tail_embedding[x.first];
		}
	}

	if (!Opt::is_diag)
	{
		Util::MatVecProd(p_head_right_mat[r], head_embedding, Opt::head_relat_rank, Opt::embeding_size, Qh_h, false);
		Util::MatVecProd(p_head_left_mat[r], Qh_h, Opt::embeding_size, Opt::head_relat_rank, off_vec, false);
	}
	else
	{
		memset(off_vec, 0, sizeof(real)* Opt::embeding_size);
		for (auto x : head_diag_mat_ele[r])
			off_vec[x.first] = x.second * head_embedding[x.first];
	}
	Util::MatPlusMat(off_vec, relat_embedding, -1, Opt::embeding_size, 1);
	if (Opt::use_tail_mat)
		Util::MatPlusMat(off_vec, PtQt_t, -1, Opt::embeding_size, 1);
	else
		Util::MatPlusMat(off_vec, tail_embedding, -1, Opt::embeding_size, 1);

	return Util::InnerProduct(off_vec, off_vec, Opt::embeding_size);
}


inline void CRelation::ConstrainParameters(int r)
{
	if (Opt::act_relat)
		for (int i = 0; i < Opt::embeding_size; ++i)
			p_relat_act_emb[r][i] = 2 * Util::Sigmoid(p_relat_emb[r][i]) - 1;
	if (Opt::act_relat_mat)
	{
		//TODO
	}
}

real CRelation::ComputeScore(int head, int r, int tail, real* Qh_h, real* Qt_t, real* off_vec)
{
	real* head_embedding = WordParams::p_embedding[head];
	real* tail_embedding = WordParams::p_embedding[tail];
	real* relat_embedding = Opt::act_relat ? p_relat_act_emb[r] : p_relat_emb[r];
	
	if (Opt::update_mat && !Opt::is_diag)
		Util::MatVecProd(p_head_right_mat[r], head_embedding, Opt::head_relat_rank, Opt::embeding_size, Qh_h, false);
	
	real PtQt_t[MAX_EMBEDDING_SIZE];

	if (Opt::use_tail_mat)
	{
		if (!Opt::is_diag)
		{
			Util::MatVecProd(p_tail_right_mat[r], tail_embedding, Opt::tail_relat_rank, Opt::embeding_size, Qt_t, false);
			Util::MatVecProd(p_tail_left_mat[r], Qt_t, Opt::embeding_size, Opt::tail_relat_rank, PtQt_t, false);
		}
		else
		{
			memset(PtQt_t, 0, sizeof(real)* Opt::embeding_size);
			for (auto x : tail_diag_mat_ele[r])
				PtQt_t[x.first] = x.second * tail_embedding[x.first];
		}
	}

	if (Opt::is_diag)
	{
		memset(off_vec, 0, sizeof(real)* Opt::embeding_size);
		for (auto x : head_diag_mat_ele[r])
			off_vec[x.first] = x.second * head_embedding[x.first];
	}
	else
	{
		if (Opt::update_mat)
			Util::MatVecProd(p_head_left_mat[r], Qh_h, Opt::embeding_size, Opt::head_relat_rank, off_vec, false);
		else
		{
			memcpy(off_vec, head_embedding, sizeof(real)* Opt::head_relat_rank);
			for (int d = Opt::head_relat_rank; d < Opt::embeding_size; ++d)
				off_vec[d] = 0;
		}
	}

	Util::MatPlusMat(off_vec, relat_embedding, -1, Opt::embeding_size, 1);
	if (Opt::use_tail_mat)
		Util::MatPlusMat(off_vec, PtQt_t, -1, Opt::embeding_size, 1);
	else
		Util::MatPlusMat(off_vec, tail_embedding, -1, Opt::embeding_size, 1);

	return Util::InnerProduct(off_vec, off_vec, Opt::embeding_size);
}


//All the *_grad is the gradient of pos_score - neg_score w.r.t. the parameter *
void CRelation::TrainRelatTriple(int head, bool head_is_word, int r, int tail)
{
	real head_grads[MAX_EMBEDDING_SIZE], tail_grads[MAX_EMBEDDING_SIZE], grads_tmp[MAX_EMBEDDING_SIZE], relat_grads[MAX_EMBEDDING_SIZE], negh_grads[MAX_EMBEDDING_SIZE], negt_grads[MAX_EMBEDDING_SIZE], negr_grads[MAX_EMBEDDING_SIZE];
	real head_left_mat_grads[MAX_EMBEDDING_SIZE * MAX_RELAT_RANK], head_right_mat_grads[MAX_EMBEDDING_SIZE * MAX_RELAT_RANK], tail_left_mat_grads[MAX_EMBEDDING_SIZE * MAX_RELAT_RANK], tail_right_mat_grads[MAX_EMBEDDING_SIZE * MAX_RELAT_RANK];
	real negr_head_left_mat_grads[MAX_EMBEDDING_SIZE * MAX_RELAT_RANK], negr_head_right_mat_grads[MAX_EMBEDDING_SIZE * MAX_RELAT_RANK], negr_tail_left_mat_grads[MAX_EMBEDDING_SIZE * MAX_RELAT_RANK], negr_tail_right_mat_grads[MAX_EMBEDDING_SIZE * MAX_RELAT_RANK];
	
	real head_mat_grads[MAX_RELAT_RANK], tail_mat_grads[MAX_RELAT_RANK]; 
	real negr_head_mat_grads[MAX_RELAT_RANK], negr_tail_mat_grads[MAX_RELAT_RANK]; //Used for the diag case

	real off_vec[MAX_EMBEDDING_SIZE], neg_off_vec[MAX_EMBEDDING_SIZE];
	real Qh_h[MAX_RELAT_RANK], Qh_negh[MAX_RELAT_RANK], Qt_t[MAX_RELAT_RANK], Qt_negt[MAX_RELAT_RANK];
	real PhT_offvec[MAX_RELAT_RANK], PtT_offvec[MAX_RELAT_RANK];
	
	memset(head_grads, 0, sizeof(real)* Opt::embeding_size);
	memset(tail_grads, 0, sizeof(real)* Opt::embeding_size);
	memset(relat_grads, 0, sizeof(real)* Opt::embeding_size);
	memset(negh_grads, 0, sizeof(real)* Opt::embeding_size);
	memset(negt_grads, 0, sizeof(real)* Opt::embeding_size);
	memset(negr_grads, 0, sizeof(real)* Opt::embeding_size);

	if (Opt::update_mat && !Opt::is_diag)
	{
		memset(head_left_mat_grads, 0, sizeof(real)* Opt::head_relat_rank * Opt::embeding_size);
		memset(head_right_mat_grads, 0, sizeof(real)* Opt::head_relat_rank * Opt::embeding_size);

		if (Opt::use_tail_mat)
		{
			memset(tail_left_mat_grads, 0, sizeof(real)* Opt::embeding_size * Opt::tail_relat_rank);
			memset(tail_right_mat_grads, 0, sizeof(real)* Opt::tail_relat_rank * Opt::embeding_size);
		}

		memset(negr_head_left_mat_grads, 0, sizeof(real)* Opt::embeding_size * Opt::head_relat_rank);
		memset(negr_head_right_mat_grads, 0, sizeof(real)* Opt::embeding_size * Opt::head_relat_rank);

		if (Opt::use_tail_mat)
		{
			memset(negr_tail_left_mat_grads, 0, sizeof(real)* Opt::embeding_size * Opt::tail_relat_rank);
			memset(negr_tail_right_mat_grads, 0, sizeof(real)* Opt::embeding_size * Opt::tail_relat_rank);
		}
	}

	if (Opt::update_mat && Opt::is_diag)
	{
		memset(head_mat_grads, 0, sizeof(real)* Opt::embeding_size);
		memset(tail_mat_grads, 0, sizeof(real)* Opt::embeding_size);
		memset(negr_head_mat_grads, 0, sizeof(real)* Opt::embeding_size);
		memset(negr_tail_mat_grads, 0, sizeof(real)* Opt::embeding_size);
	}

	//Sample neg head word and neg tail word
	int neg_head, neg_tail, neg_r;
	do
	{
		neg_head = SampleWordIdx();
	} while (neg_head == head || neg_head == tail);

	do
	{
		neg_tail = SampleWordIdx();
	} while (neg_tail == head || neg_tail == tail);

	do
	{
		neg_r = SampleRelatIdx();
	} while (neg_r == r);

	real* head_embedding = WordParams::p_embedding[head];
	real* tail_embedding = WordParams::p_embedding[tail];
	real* relat_embedding = p_relat_emb[r];
	real* relat_act_embedding = Opt::act_relat ? p_relat_act_emb[r] : NULL;

	real* neg_head_embedding = WordParams::p_embedding[neg_head];
	real* neg_tail_embedding = WordParams::p_embedding[neg_tail];
	real* neg_r_embedding = p_relat_emb[neg_r];
	real* neg_r_act_embedding = Opt::act_relat ? p_relat_act_emb[neg_r] : NULL;

	real pos_score = ComputeScore(head, r, tail, Qh_h, Qt_t, off_vec); 
	real negh_score = ComputeScore(neg_head, r, tail, Qh_negh, Qt_t, neg_off_vec);
	
	real hgap;
	bool is_negh_margin_satisfied;

	hgap = negh_score - pos_score;
	is_negh_margin_satisfied = hgap > Opt::margin;
	if (!is_negh_margin_satisfied) //margin is not satisfied
		ComputeGradient(1, Opt::relat_neg_weight, r, grads_tmp, negh_grads, tail_grads, relat_grads,
			head_left_mat_grads, head_right_mat_grads, tail_left_mat_grads, tail_right_mat_grads,
			neg_off_vec, Qh_negh, Qt_t, PhT_offvec, PtT_offvec, neg_head_embedding, tail_embedding,
			head_mat_grads, tail_mat_grads);
	
	//Begin updating the loss and grads for ||LR(h - t)||_2^2 - ||LR(h - negt)||_2^2
	real negt_score = ComputeScore(head, r, neg_tail, Qh_h, Qt_negt, neg_off_vec);
	real tgap;
	bool is_negt_margin_satisfied;

	tgap = negt_score - pos_score;
	is_negt_margin_satisfied = tgap > Opt::margin;
	if (!is_negt_margin_satisfied)
		ComputeGradient(1, Opt::relat_neg_weight, r, grads_tmp, head_grads, negt_grads, relat_grads,
			head_left_mat_grads, head_right_mat_grads, tail_left_mat_grads, tail_right_mat_grads, neg_off_vec,
			Qh_h, Qt_negt, PhT_offvec, PtT_offvec, head_embedding, neg_tail_embedding,
			head_mat_grads, tail_mat_grads);
	
	real negr_score = ComputeScore(head, neg_r, tail, Qh_negh, Qt_negt, neg_off_vec);
	real rgap;
	bool is_negr_margin_satisfied = true;;
	
	rgap = negr_score - pos_score;
	is_negr_margin_satisfied = rgap > Opt::margin;
	if (!is_negr_margin_satisfied)
		ComputeGradient(1, Opt::relat_neg_weight, neg_r, grads_tmp, head_grads, tail_grads, negr_grads,
			negr_head_left_mat_grads, negr_head_right_mat_grads, negr_tail_left_mat_grads, negr_tail_right_mat_grads, neg_off_vec,
			Qh_negh, Qt_negt, PhT_offvec, PtT_offvec, head_embedding, tail_embedding,
			negr_head_mat_grads, negr_tail_mat_grads);
	
	if (!Opt::sig_relat && is_negh_margin_satisfied && is_negt_margin_satisfied && is_negr_margin_satisfied)
		return;

	int effect_cnt = (is_negh_margin_satisfied ? 0 : 1) + (is_negt_margin_satisfied ? 0 : 1) + (is_negr_margin_satisfied ? 0 : 1);
	ComputeGradient(-1, effect_cnt, r, grads_tmp, head_grads, tail_grads, relat_grads,
		head_left_mat_grads, head_right_mat_grads, tail_left_mat_grads, tail_right_mat_grads, off_vec,
		Qh_h, Qt_t, PhT_offvec, PtT_offvec, head_embedding, tail_embedding,
		head_mat_grads, tail_mat_grads);

	//Gradient Checking
	/*real gap = (is_negh_margin_satisfied ? 0 : negh_score - pos_score) + (is_negt_margin_satisfied ? 0 : negt_score - pos_score) + (is_negr_margin_satisfied ? 0 : negr_score - pos_score);

	
	if (rand() < 20)
	{
		const double epsilon = 1e-6;
		//head_embedding[idx] += epsilon;
		//p_head_left_mat[r][idx] += epsilon;
		//p_tail_right_mat[r][idx] += epsilon;
		//tail_embedding[idx] += epsilon;
		//p_actual_left_mat[r][idx] = 2 * Util::Sigmoid(p_left_mat[r][41]) - 1;
		//p_tail_left_mat[r][idx] += epsilon;
		//p_head_left_mat[neg_r][idx] += epsilon;
		//p_actual_right_mat[r][idx] = 2 * Util::Sigmoid(p_right_mat[r][idx]) - 1;
		//p_relat_emb[neg_r][idx] += epsilon;
		//p_relat_act_emb[neg_r][idx] = 2 * Util::Sigmoid(p_relat_emb[neg_r][idx]) - 1;

		idx = -1;
		for (auto x : tail_diag_mat_ele[r])
			idx = x.first;

		printf("\n");

		tail_diag_mat_ele[neg_r][idx] += epsilon;
		
		real new_gap = 0, pos_score = ComputeLoss(head, tail, r);
		//ComputeLoss(head, tail, neg_r) - ComputeLoss(head, tail, r);
		real neg_gap = ComputeLoss(neg_head, tail, r) - pos_score;
		if (neg_gap <= Opt::margin)
			new_gap += neg_gap;
		neg_gap = ComputeLoss(head, neg_tail, r) - pos_score;
		if (neg_gap <= Opt::margin)
			new_gap += neg_gap;
		neg_gap = ComputeLoss(head, tail, neg_r) - pos_score;
		if (neg_gap <= Opt::margin)
			new_gap += neg_gap;

		printf("real gradient: %.5f, our gradient %.5f, idx:%d\n", (new_gap - gap) / epsilon, negr_tail_mat_grads[idx], idx);
		//p_tail_right_mat[r][idx] -= epsilon;
		//p_head_left_mat[neg_r][idx] -= epsilon;
		//tail_embedding[idx] -= epsilon;
		//p_actual_right_mat[r][idx] = 2 * Util::Sigmoid(p_right_mat[r][idx]) - 1;
		//head_embedding[idx] -= epsilon;
		//p_relat_emb[neg_r][idx] -= epsilon;
		//p_relat_act_emb[neg_r][idx] = 2 * Util::Sigmoid(p_relat_emb[neg_r][idx]) - 1;
		//p_actual_left_mat[neg_r][idx] = 2 * Util::Sigmoid(p_left_mat[r][41]) - 1;
		tail_diag_mat_ele[neg_r][idx] -= epsilon;
	}*/

	double step_size = GetStepSize(head, r, tail);
	step_size /= effect_cnt;

	Util::MatPlusMat(relat_embedding, relat_grads, step_size, Opt::embeding_size, 1);
	
	Util::MatPlusMat(head_embedding, head_grads, step_size, Opt::embeding_size, 1);
	
	Util::MatPlusMat(tail_embedding, tail_grads, step_size, Opt::embeding_size, 1);

	if (!is_negt_margin_satisfied)
		Util::MatPlusMat(neg_tail_embedding, negt_grads, step_size, Opt::embeding_size, 1);

	if (!is_negh_margin_satisfied)
		Util::MatPlusMat(neg_head_embedding, negh_grads, step_size, Opt::embeding_size, 1);

	if (!is_negr_margin_satisfied)
		Util::MatPlusMat(neg_r_embedding, negr_grads, step_size, Opt::embeding_size, 1);

	if (Opt::update_mat && !Opt::is_diag)
	{
		Util::MatPlusMat(p_head_left_mat[r], head_left_mat_grads, step_size, Opt::embeding_size, Opt::head_relat_rank);
		Util::MatPlusMat(p_head_right_mat[r], head_right_mat_grads, step_size, Opt::head_relat_rank, Opt::embeding_size);
		if (Opt::use_tail_mat)
		{
			Util::MatPlusMat(p_tail_left_mat[r], tail_left_mat_grads, step_size, Opt::embeding_size, Opt::tail_relat_rank);
			Util::MatPlusMat(p_tail_right_mat[r], tail_right_mat_grads, step_size, Opt::tail_relat_rank, Opt::embeding_size);
		}

		if (!is_negr_margin_satisfied)
		{
			Util::MatPlusMat(p_head_left_mat[neg_r], negr_head_left_mat_grads, step_size, Opt::embeding_size, Opt::head_relat_rank);
			Util::MatPlusMat(p_head_right_mat[neg_r], negr_head_right_mat_grads, step_size, Opt::embeding_size, Opt::head_relat_rank);
			if (Opt::use_tail_mat)
			{
				Util::MatPlusMat(p_tail_left_mat[neg_r], negr_tail_left_mat_grads, step_size, Opt::embeding_size, Opt::tail_relat_rank);
				Util::MatPlusMat(p_tail_right_mat[neg_r], negr_tail_right_mat_grads, step_size, Opt::embeding_size, Opt::tail_relat_rank);
			}
		}
	}
	else if (Opt::update_mat && Opt::is_diag)
	{
		for (auto x : head_diag_mat_ele[r])
			head_diag_mat_ele[r][x.first] += step_size *  head_mat_grads[x.first];
		if (Opt::use_tail_mat)
			for (auto x : tail_diag_mat_ele[r])
				tail_diag_mat_ele[r][x.first] += step_size * tail_mat_grads[x.first];
		if (!is_negr_margin_satisfied)
		{
			for (auto x : head_diag_mat_ele[neg_r])
				head_diag_mat_ele[neg_r][x.first] += step_size *  negr_head_mat_grads[x.first];
			if (Opt::use_tail_mat)
				for (auto x : tail_diag_mat_ele[neg_r])
					tail_diag_mat_ele[neg_r][x.first] += step_size * negr_tail_mat_grads[x.first];
		}
	}

	ConstrainParameters(r);
	if (!is_negr_margin_satisfied)
		ConstrainParameters(neg_r);
}

//Compute the grad w.r.t to ||LR(h-t)||_2^2 or -||LR(negh-t)||_2^2 or -||LR(h-negt)||_2^2
void CRelation::ComputeGradient(int label, real weight, int relat, real* grads_tmp, real* head_grads, real* tail_grads, real* relat_grads,
	real* head_left_mat_grads, real* head_right_mat_grads, real* tail_left_mat_grads, real* tail_right_mat_grads, 
	real* off_vec, real* Qh_h, real* Qt_t, real* PhT_off_vec, real* PtT_off_vec, real* head_embedding, real* tail_embedding,
	real* diag_head_mat_grads, real* diag_tail_mat_grads)
{
	for (int i = 0; i < Opt::embeding_size; ++i)
		relat_grads[i] -= 2 * label * weight * off_vec[i] * (Opt::act_relat ? Util::d2Sigmoid(p_relat_act_emb[relat][i]) : 1);

	if (!Opt::is_diag)
	{
		if (Opt::update_mat)
		{
			Util::MatVecProd(p_head_left_mat[relat], off_vec, Opt::embeding_size, Opt::head_relat_rank, PhT_off_vec, true); //Get the gradient w.r.t to head_embeddings
			Util::MatVecProd(p_head_right_mat[relat], PhT_off_vec, Opt::head_relat_rank, Opt::embeding_size, grads_tmp, true);
			Util::MatPlusMat(head_grads, grads_tmp, 2 * label * weight, Opt::embeding_size, 1);
		}
		else
			Util::MatPlusMat(head_grads, off_vec, 2 * label * weight, Opt::embeding_size, 1);

		if (Opt::use_tail_mat)
		{
			Util::MatVecProd(p_tail_left_mat[relat], off_vec, Opt::embeding_size, Opt::tail_relat_rank, PtT_off_vec, true);
			Util::MatVecProd(p_tail_right_mat[relat], PtT_off_vec, Opt::tail_relat_rank, Opt::embeding_size, grads_tmp, true);
			Util::MatPlusMat(tail_grads, grads_tmp, -2 * label * weight, Opt::embeding_size, 1); //Get the gradient w.r.t to tail_embeddings
		}
		else
			Util::MatPlusMat(tail_grads, off_vec, -2 * label * weight, Opt::embeding_size, 1);
	}
	else
	{
		for (auto x : head_diag_mat_ele[relat])
			head_grads[x.first] += 2 * label * weight * x.second * off_vec[x.first];
		if (Opt::use_tail_mat)
		{
			for (auto x : tail_diag_mat_ele[relat])
				tail_grads[x.first] -= 2 * label * weight * x.second * off_vec[x.first];
		}
		else
			Util::MatPlusMat(tail_grads, off_vec, -2 * label * weight, Opt::embeding_size, 1);
	}
	if (!Opt::update_mat)
		return;

	if (!Opt::is_diag)
	{
		for (int i = 0; i < Opt::head_relat_rank; ++i)
			for (int j = 0; j < Opt::embeding_size; ++j)
				head_right_mat_grads[i * Opt::embeding_size + j] += 2 * label * weight * PhT_off_vec[i] * head_embedding[j];

		if (Opt::use_tail_mat)
		{
			for (int i = 0; i < Opt::tail_relat_rank; ++i)
				for (int j = 0; j < Opt::embeding_size; ++j)
				tail_right_mat_grads[i * Opt::embeding_size + j] -= 2 * label * weight * PtT_off_vec[i] * tail_embedding[j];
		}

		for (int i = 0; i < Opt::embeding_size; ++i)
			for (int j = 0; j < Opt::head_relat_rank; ++j)
				head_left_mat_grads[i * Opt::head_relat_rank + j] += 2 * label * weight * off_vec[i] * Qh_h[j];
			
		if (Opt::use_tail_mat)
		{
			for (int i = 0; i < Opt::embeding_size; ++i)
				for (int j = 0; j < Opt::head_relat_rank; ++j)
					tail_left_mat_grads[i * Opt::tail_relat_rank + j] -= 2 * label * weight * off_vec[i] * Qt_t[j];
		}
	}
	else
	{
		for (auto x : head_diag_mat_ele[relat])
			diag_head_mat_grads[x.first] += 2 * label * weight * off_vec[x.first] * head_embedding[x.first];
		if (Opt::use_tail_mat)
			for (auto x : tail_diag_mat_ele[relat])
				diag_tail_mat_grads[x.first] -= 2 * label * weight * off_vec[x.first] * tail_embedding[x.first];
	}
}

real CRelation::SeqTrainRelation()
{
	if (know_lock == true)
		return 0;
	know_lock = true;
	real b_result = 0, b_false_result = 0;

	real curr_lr = NN::learning_rate;
	int total_update_cnt = (int)(1.0 / Opt::know_update_per_progress);

	//NN::learning_rate = Opt::init_learning_rate * (1 - update_know_cnt / (real)total_update_cnt);
	NN::learning_rate = Opt::init_learning_rate;
	if (NN::learning_rate < eps)
	{
		NN::learning_rate = curr_lr;
		know_lock = false;
		return 0;
	}
	
	for (int i = 0; i < list.size(); ++i)
	{
		b_result += ComputeLoss(list[i].w1, list[i].w2, list[i].r);
		b_false_result += ComputeLoss(SampleWordIdx(), list[i].w2, list[i].r);
	}

	srand(time(NULL));
	std::vector<int> seqs;
	for (int i = 0; i < list.size(); ++i)
		seqs.push_back(i);

	for (int j = 0; j < Opt::know_iter; ++j)
	{
		std::random_shuffle(seqs.begin(), seqs.end());
		for (int i = 0; i < list.size(); ++i)
			TrainRelatTriple(list[seqs[i]].w1, true, list[seqs[i]].r, list[seqs[i]].w2);
	}
	NN::learning_rate = curr_lr;
	update_know_cnt++;
	
	real a_result = 0, a_false_result = 0;

	if (rand() / (RAND_MAX + 1.0) < 0.25)
	{
		for (int i = 0; i < list.size(); ++i)
		{
			a_result += ComputeLoss(list[i].w1, list[i].w2, list[i].r);
			a_false_result += ComputeLoss(SampleWordIdx(), list[i].w2, list[i].r);
		}
		std::cout << "\nid: " << std::this_thread::get_id();
		printf(" %dth update, %.5f", update_know_cnt + 1, NN::learning_rate);
		printf("\nBefore loss: %.5f\tAfter loss: %.5f\t", b_result / list.size(), a_result / list.size());
		printf("Before cha: %.5f\tAfter cha: %.5f\n", (b_false_result - b_result) / list.size(), (-a_result + a_false_result) / list.size());
	}
	know_lock = false;
	return Opt::lambda * a_result;
}

double inline CRelation::GetStepSize(int h, int r, int t)
{
	double step_size = NN::learning_rate * Opt::lambda;
	if (Opt::know_update_per_progress > eps)
	{
		step_size *= list.size();
		step_size /= (relat_size * relat_triple_size[r]);
	}
	else if (false)
	{
		step_size *= g_reader.total_words;
		step_size /= (g_dict->GetWordInfo(h)->freq + g_dict->GetWordInfo(t)->freq);
		step_size /= (relat_size * relat_triple_size[r]);
	}
	return step_size;
}