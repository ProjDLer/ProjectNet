#pragma once

#include <concurrent_unordered_map.h>
#include <string>
#include <vector>

const int MAX_WORD_SIZE = 901;

struct WordInfo
{
	std::string word;
	long long freq;
	WordInfo()
	{
		freq = 0;
		word.clear();
	}
	WordInfo(const std::string& _word, long long _freq)
	{
		word = _word;
		freq = _freq;
	}
};

class Dictionary
{
public:
	Dictionary();
	void Clear();
	void RemoveWordsLessThan(long long min_count);
	void MergeInfrequentWords(long long threshold);
	void Insert(const char* word, long long cnt = 1);
	int GetWordIdx(const char* word);
	const WordInfo* GetWordInfo(const char* word);
	const WordInfo* GetWordInfo(int word_idx);
	int Size();

private:
	std::vector<WordInfo> m_word_info;
	Concurrency::concurrent_unordered_map<std::string, int> m_word_idx_map;
};