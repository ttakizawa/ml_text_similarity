# # -*- coding: utf-8 -*-
import os
import re
import csv
import numpy as np
import pandas as pd
import MeCab

DEFAULT_PARSE_LOG_PATH = "parse_text_log.csv"
PARSE_LOG_HEADER = []

def output_log(log_text,path):
    if os.path.exists(path) and os.path.isfile(path):
        print(log_text)
        with open(path,'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            for l in log_text:
                writer.writerow(l)
    else:
        with open(path,'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(PARSE_LOG_HEADER)
            for l in log_text:
                writer.writerow(l)


def clean_text(text):
    re_hiragana = re.compile(r'[\u3041-\u3093]')#[ぁ-んー－]
    re_mark = re.compile(r'[0123456789０１２３４５６７８９！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？、。・,./『』【】「」→←○]+')
    result = text.replace("\u3000","").replace("\n","")
    result = re.sub(re_mark, "", result)
    if re_hiragana.match(result) and len(result):
        result = ''
    return result

class Parser:
    def __init__(self):
        self.tagger = MeCab.Tagger("-Ochasen")
        self.tagger.parse('')

    def parse_text(self, text, log=False, path=DEFAULT_PARSE_LOG_PATH):
        word_list = []
        log_text = []

        if text == '' or text is None:
            word_list = ['']
            return word_list
        re_hiragana = re.compile(r'[\u3041-\u3093]')#[ぁ-んー－]
        node = self.tagger.parseToNode(text)
        while node:
            if log:
                print(node.surface,node.feature)
                log_text.append([node.surface] + node.feature.split(","))

            add_flag = False
            f = node.feature.split(",")
            #if node.feature.startswith('名詞') or node.feature.startswith('動詞') or node.feature.startswith('形容詞') :
            if f[0] == '名詞' and f[1] != '数':
                add_flag = True
            elif f[0] in ['形容詞','動詞']:
                add_flag = True
            add_text = clean_text(node.surface)

            if add_flag and add_text != '':
                word_list.append(add_text)
            node = node.next

        if log:
            output_log(log_text,path)

        if word_list == False:
            word_list = ['']
            return word_list
        return word_list


class Text_Vectornizer(object):
    def __init__(self,model=None,matrix_size=100,parser=None):
        self.model = model
        self.matrix_size = matrix_size
        if parser is None:
            self.parser = Parser()
        else:
            self.parser = parser


    def transform(self,text_list):
        if self.model is not None:
            X = [self.text2vector(text) for text in text_list]
        else:
            X = None
        return X


    def text2vector(self,text):
        vec_text = np.zeros(self.model.vector_size)
        word_list = parse_text(text)
        for word in word_list:
            if word in self.model.wv:
                vec_text += self.model.wv[word]
        return vec_text


    def text2matrix(self, text, reverse=False):
        matrix_text = np.zeros((self.model.vector_size,1))
        word_list = self.parser.parse_text(text)
        cnt = 0
        for word in word_list:
            if word in self.model.wv:
                word_vec = self.model.wv[word].reshape(self.model.vector_size,1)
                if cnt == 0:
                    matrix_text = word_vec
                else:
                    matrix_text = np.hstack((matrix_text, word_vec))
                cnt += 1
            if cnt >= self.matrix_size:
                if reverse:
                    matrix_text = matrix_text[:, ::-1]
                return matrix_text

        for i in range(self.matrix_size - cnt):
            matrix_text = np.hstack((matrix_text, np.zeros((self.model.vector_size,1))))
        if reverse:
            matrix_text = matrix_text[:, ::-1]
        return matrix_text
