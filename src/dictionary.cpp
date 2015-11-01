#include "Dictionary.h"

Dictionary::Dictionary()
{
	m_word_idx_map.clear();
	m_word_info.clear();
}

void Dictionary::Clear()
{
	m_word_idx_map.clear();
	m_word_info.clear();
}

void Dictionary::MergeInfrequentWords(long long threshold)
{
	m_word_idx_map.clear();
	std::vector<WordInfo> tmp_info;
	tmp_info.clear();
	int infreq_idx = -1;

	for (auto& word_info : m_word_info)
	{
		if (word_info.freq >= threshold || word_info.freq == 0)
		{
			m_word_idx_map[word_info.word] = (int)tmp_info.size();
			tmp_info.push_back(word_info);
		}
		else {
			if (infreq_idx < 0)
			{
				WordInfo infreq_word_info;
				infreq_word_info.word = "WE_ARE_THE_INFREQUENT_WORDS";
				infreq_word_info.freq = 0;
				m_word_idx_map[infreq_word_info.word] = (int)tmp_info.size();
				infreq_idx = (int)tmp_info.size();
				tmp_info.push_back(infreq_word_info);
			}
			m_word_idx_map[word_info.word] = infreq_idx;
			tmp_info[infreq_idx].freq += word_info.freq;
		}
	}
	m_word_info = tmp_info;
}

void Dictionary::RemoveWordsLessThan(long long min_count)
{
	m_word_idx_map.clear();
	std::vector<WordInfo> tmp_info;
	tmp_info.clear();
	for (auto& info : m_word_info)
	{
		if (info.freq >= min_count || info.freq == 0)
		{
			m_word_idx_map[info.word] = (int)tmp_info.size();
			tmp_info.push_back(info);
		}
	}
	m_word_info = tmp_info;
}

void Dictionary::Insert(const char* word, long long cnt)
{
	auto& it = m_word_idx_map.find(word);
	if (it != m_word_idx_map.end())
		m_word_info[it->second].freq += cnt;
	else 
	{
		m_word_idx_map[word] = (int)m_word_info.size();
		m_word_info.push_back(WordInfo(word, cnt));
	}
}

int Dictionary::GetWordIdx(const char* word)
{
	auto& it = m_word_idx_map.find(word);
	if (it != m_word_idx_map.end())
		return it->second;
	return -1;
}

int Dictionary::Size()
{
	return (int)m_word_info.size();
}

const WordInfo* Dictionary::GetWordInfo(const char* word)
{
	auto& it = m_word_idx_map.find(word);
	if (it != m_word_idx_map.end())
		return GetWordInfo(it->second);
	return NULL;
}
	
const WordInfo* Dictionary::GetWordInfo(int word_idx)
{
	if (word_idx >= 0 && word_idx < m_word_info.size())
		return &m_word_info[word_idx];
	return NULL;
}