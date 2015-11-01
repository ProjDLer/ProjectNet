#include "wordparams.h"
#include "neuralnet.h"
#include <algorithm>
#include <map>
#include "relation.h"
#include <set>

int* WordParams::relat_cnt = NULL;
real** WordParams::p_embedding = NULL;
real** WordParams::p_relat_prior = NULL;
int** WordParams::p_relats = NULL;

extern Dictionary* g_dict;
extern CRelation g_relat;
extern FILE* g_debug;

void WordParams::InitRelatCnt()
{
	relat_cnt = (int*)calloc(g_dict->Size(), sizeof(int));
	NN::embedding_cnt = g_dict->Size();
	for (int i = 0; i < g_dict->Size(); ++i)
		relat_cnt[i] = 1;
	if (!Opt::use_relation || !Opt::use_know_in_text)
	{
		printf("Totally %d input embeddings\n", NN::embedding_cnt);
		return;
	}

	int start, end, r;
	std::set<int> relat_set;
	p_relats = (int**)malloc(g_dict->Size() * sizeof(int*));

	for (auto& word_map : g_relat.entityid2wordid_table)
	{
		relat_set.clear();
		int word_id = word_map.second;
		auto word_loc = g_relat.head_idx_range_table.find(word_id);
		if (word_loc != g_relat.head_idx_range_table.end())
		{
			start = word_loc->second.first, end = word_loc->second.second;
			for (int i = start; i < end; ++i)
			{
				r = g_relat.list[i].r;
				if (relat_set.find(r) == relat_set.end())
					relat_set.insert(r);
			}
		}

		word_loc = g_relat.tail_idx_range_table.find(word_id);
		if (word_loc != g_relat.tail_idx_range_table.end())
		{
			start = word_loc->second.first; end = word_loc->second.second;
			for (int i = start; i < end; ++i)
			{
				r = g_relat.list_reverse[i].r;
				if (relat_set.find(r) == relat_set.end())
					relat_set.insert(r);
			}
		}
		relat_cnt[word_id] += (int)relat_set.size();
		p_relats[word_id] = (int*)malloc((relat_cnt[word_id] - 1) * sizeof(int));
		NN::embedding_cnt += (int)relat_set.size();

		int i = 0;
		for (auto& x : relat_set)
			p_relats[word_id][i++] = x;
	}
	printf("Totally %d input embeddings\n", NN::embedding_cnt);
}

void WordParams::InitWordParams()
{
	WordParams::p_embedding = (real**)malloc(g_dict->Size() * sizeof(real*));
	WordParams::p_relat_prior = (real**)malloc(g_dict->Size() * sizeof(real*));
	long long cnt = 0;
	for (int i = 0; i < g_dict->Size(); ++i)
	{
		WordParams::p_embedding[i] = &NN::input_embedding[cnt * Opt::embeding_size];
		WordParams::p_relat_prior[i] = &NN::relat_prior[cnt];
		for (int j = 0; j < WordParams::relat_cnt[i]; ++j)
			WordParams::p_relat_prior[i][j] = (real)1.0 / WordParams::relat_cnt[i];
		cnt += WordParams::relat_cnt[i];
	}
}
