import numpy as np
from numpy import linalg
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import torch
from random import sample
import re
import pickle 
import json
import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
import transformers
import os
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2TokenizerFast, BertTokenizer, BertModel, logging

import argparse

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


###########################################################################################
# # model_name = "google/flan-t5-small"
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# #model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
# #model = model.to(device)


# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_name,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )



# def get_completion(msg_in):

#     messages = [
#         {
#             "role": "user",
#             "content": "You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic. Do not generate examples in your answer.",
#         },
#         {
#             "role":"assistant",
#             "content": "I understand.",
#         },
#         {
#             "role": "user", 
#             "content": msg_in,
#         }
#     ]

#     #prompt = msg_in    
#     prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10, top_p=1.0)

#     out_text = []
#     for x in range(0, 10):
#         out_text.append(outputs[x]["generated_text"])
#     return out_text




#####################################################################################################
system_message = """The following is a conversation between a Human and an AI Assistant.
The assistant is helpful, respectful and honest, and it always answers as helpfully as possible, while being safe.
The Assistant's answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that the Assistant's responses are socially unbiased and positive in nature.
If a question by the human does not make any sense, or is not factually coherent, the Assistant should explain why instead of answering something not correct.
If the Assistant does not know the answer to a question, please don't share false information.
Solve the given problem step by step following the examples
####

"""
api_keys = ["EMPTY"]
endpoint_urls = ["https://ca4f-130-75-152-24.ngrok.app"]
llm_names = []

for api_key, endpoint_url in zip(api_keys, endpoint_urls):
    if 'hf.space' in endpoint_url:
        model_name = endpoint_url.replace('https://', '').replace('.hf.space', '').replace('/', '')
    else:
        openai.api_key = api_key
        openai.api_base = f"{endpoint_url}/v1"
        model_names = openai.Model.list()
        model_name = model_names["data"][0]["id"]
    llm_names.append(model_name)


# Gen response from API
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def get_completion(prompt, api_key, endpoint_url, hard_code_exception=False):

    max_tokens=256
    if 'hf.space' in endpoint_url:
        client = Client(endpoint_url)
        result = client.predict(
                        prompt, # str in 'Message' Textbox component
                        api_name="/chat"
        )
        return result.strip()
    openai.api_key = api_key
    openai.api_base = f"{endpoint_url}/v1"
    model_names = openai.Model.list()
    model_name = model_names["data"][0]["id"]

    res = openai.Completion.create(
        model=model_name,  # Replace with your model name
        prompt=system_message + prompt,
        # messages=[
        #     {"role": "system", "content": system_message},
        #     {"role": "user", "content": prompt},
        # ],
        temperature=0.25,
        top_k=10,
        top_p=1.0,
        n=10,
        max_tokens=256,
    )
    out_text = []
    for x in range(0, 10):
        if res['choices'][x]['text'] == "":
            continue
        else:
            return res['choices'][x]['text'].strip()
    return out_text
###############################################################################################



def llm_output(user_query, hard_code_exception=False):
    results = get_completion(user_query, api_keys[0], endpoint_urls[0], hard_code_exception=hard_code_exception)
    #results = get_completion(user_query)
    
    return results


# Self consistency on 10 generated answers
def self_con(tmp_list):
    ans_list = []
    for tmp in tmp_list:
        print("tmp",tmp)
        ans = ""
        if len(tmp.split("Final Answer:"))>0:
            ans = tmp.split("Final Answer:")[-1]
            ans = ans.split("\n")[0]
            # print(ans)
            if "each" in ans:  ans = ans.split("each")[0]
            if "=" in ans: ans = ans.split("=")[-1]
            ans = re.sub(r'[^0-9.]',"",ans)
            if len(ans)>0 and ans[-1]==".": ans = ans[:-1]
            # print(ans, "**************")
            try:
                float(ans)
                ans = round(float(ans))
                ans_list.append(ans)
            except: pass
        # ans_list.append(ans)

    # print(ans_list)
    d = {}
    for i in ans_list:
        if i=="":
            continue
        if int(i) in d:
            d[int(i)] += 1
        else:
            d[int(i)] = 1
    # print(d)
    n = sorted(d.items(), key=lambda x:x[1], reverse=True)
    return n

# Strip answer from sentence
def clean_ans(s):
    ans_s = s.split("#### ")[1]
    ans_s = ans_s.replace(",","")
    return ans_s

def get_prompt(ex):
    s = "\n\n"
    s += "Question:" + ex["question"]+"\n"
    ex["answer"] = re.sub("<<.*?>>", "", ex["answer"])
    ex["answer"] = ex["answer"].replace("#### ", "Final Answer:")
    s += ex["answer"]
    return s


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prompt_for_manual_prediction(ex, shots):
    #print(shots)
    prompt = "Follow given examples and solve the given Test Question at end in similar manner by giving step by step reasoning followed by the Final Answer.\n\n"
    for index, s in shots.iterrows():
        prompt += get_prompt(s)

    prompt += "\n\nFollowing the given examples think step by step to solve the problem and generate step by step reasoning strictly preceded by Answer: and generate Final Answer preceded by Final Answer: for the below question.\n\n" 
    prompt += "Question:" + ex["question"]
    
    return prompt
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def LLM_avg_error(exemplars_set, val_data):
    # stop_signal = "\n\n"
    error=[]
    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    for exemplars in tqdm(exemplars_set,total=len(exemplars_set),desc="LLM Loss Fn Exemplar"):
        matches = 0
        mismatches = 0
        exnum = 0
        # acc_records = []
        for index, row in val_data.iterrows():
            print("********************************+")
            prompt = prompt_for_manual_prediction(row, exemplars)
            tmp = llm_output(prompt)
            print("tmp_list",tmp)
    
            ground_truth = int(clean_ans(row["answer"]))

            if len(tmp.split("Final Answer:"))>0:
                ans = tmp.split("Final Answer:")[-1]
                ans = ans.split("\n")[0]
                # print(ans)
                if "each" in ans:  ans = ans.split("each")[0]
                if "=" in ans: ans = ans.split("=")[-1]
                ans = re.sub(r'[^0-9.]',"",ans)
                if len(ans)>0 and ans[-1]==".": ans = ans[:-1]
                # print(ans, "**************")
                try:
                    float(ans)
                    ans = round(float(ans))
                except: pass
        
            ground_truth = int(clean_ans(row["answer"]))

            answer = ans
            if ground_truth==answer:
                matches+=1
            else: 
                mismatches += 1           
            
            exnum+=1

        error.append(mismatches/(matches+mismatches))

    return error
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def LLM_error_indicator(exemplars, val_data):
    # stop_signal = "\n\n"
    #arm_rewards=[]
    # increment error if model predicted answer is not equal to the ground truth answer for the test question by passing the exemplars
    subset_accuracy = 0
    for index, row in val_data.iterrows():
        prompt = prompt_for_manual_prediction(row, exemplars)
        # print("*********************************")
        # print(prompt)
        # print("*********************************")
        tmp = llm_output(prompt)
        #print("********************************+",tmp_list)

        if len(tmp.split("Final Answer:"))>0:
            ans = tmp.split("Final Answer:")[-1]
            ans = ans.split("\n")[0]
            # print(ans)
            if "each" in ans:  ans = ans.split("each")[0]
            if "=" in ans: ans = ans.split("=")[-1]
            ans = re.sub(r'[^0-9.]',"",ans)
            if len(ans)>0 and ans[-1]==".": ans = ans[:-1]
            # print(ans, "**************")
            try:
                float(ans)
                ans = round(float(ans))
            except: pass
        
        ground_truth = int(clean_ans(row["answer"]))

        answer = ans
        if answer=="": subset_accuracy+= 0
        else: subset_accuracy+= 1
        print("subset_accuracy",subset_accuracy)
    averaged_val_acc = subset_accuracy/val_data.shape[0]

    return averaged_val_acc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class top_m_arm:

    def __init__(self, args, X, m, val_data=None, subsets = None, theta=None, delta=0.05, epsilon=0.11):
        self.not_terminated = 0
        self.current_n_simu = 0
        self.delta = delta
        self.epsilon = epsilon
        self.m = m
        self.val_data=val_data
        self.subsets = subsets
        self.X = X
        self.N, self.K = np.shape(X)
        self.arms = list(range(self.K))
        self.true_theta = theta
        self.theta = None
        self.T_init = 0
        self.n_das = 5
        self.sigma=0.5
        self.index_time = 0


    def reset_bandit_specific_parameters(self):
        assert np.shape(self.X)[0] == self.N and np.shape(self.X)[1] == self.K and len(np.shape(self.X)) == 2
        self.T_init = 0
        self.hyper = 1.
        self.B_inv = 1/float(self.hyper**2)*np.eye(self.N)
        self.b = np.zeros(self.N)
        self.theta = np.random.normal(0, 1, size=(1, self.N)) 


    def clear(self):
        ## Collected rewards 
        self.rewards = [] 
        ## Pulled arms
        self.pulled_arms = []
        ## Should the learning step stop?
        self.done = False
        self.t = 0
        ## Estimated empirical average reward for each arm
        self.means = np.zeros(self.K)
        self.cum_sum = [0]*self.K 
        ## Contextual algorithms 
        self.B_inv = None 
        self.b = None 
        self.theta = None 
        ## Number of times each arm has been sampled so far 
        self.na = np.zeros(self.K) 
        ## Plot arguments 
        ## Confidence intervals (if needed) 
        self.confidence_intervals = None 
        self.best_arm, self.challenger = [None]*2 
        ## Stopping quantity 
        self.B = float("inf") 
        self.indices = None 
        ## "optimal" allocation 
        self.ratio = {}  
        self.reset_bandit_specific_parameters()


    def matrix_norm(self, x, A):
        '''Mahalanobis norm/Matrix norm: sqrt(x^TAx)'''
        #print("np.shape(x))",np.shape(x),np.shape(A))
        # if(len(np.shape(x))!=2):
        #     x = x.reshape((-1, 1))
        # print("np.shape(x))",np.shape(x),np.shape(A))
        #assert (is_of_type(x, "numpy.ndarray") or is_of_type(x, "numpy.matrix")) and is_of_type(A, "numpy.matrix") or is_of_type(A, "numpy.ndarray")
        assert len(np.shape(x)) == 2 and len(np.shape(A)) == 2
        assert np.shape(x)[0] == np.shape(A)[0] and np.shape(A)[0] == np.shape(A)[1]
        return np.sqrt((x.T).dot(A).dot(x))


    def gap(self, i, args, j=None):
        keys = ["theta"]
        assert all([k in list(args.keys()) for k in keys])
        assert all([str(x) != "None" for x in keys])
        assert np.shape(args["theta"])[1] == np.shape(self.X)[0]
        assert all([0 <= x and x < np.shape(self.X)[1] for x in [i]])
        if (str(j) != "None"):
            assert all([0 <= x and x < np.shape(self.X)[1] for x in [j]])
            gap_ = np.dot(args["theta"], self.X[:, i]-self.X[:,j])
        else:
            gap_ = np.dot(args["theta"], self.X[:, i])
        return gap_

    def HeuristicBeta(self,delta):
        #assert is_of_type(delta, "float")
        assert 0 < delta and delta < 1
        return (lambda t : np.log((np.log(t)+1)/float(delta)))

    def variance(self, i, t, args, j=None):
        self.beta= self.HeuristicBeta(self.delta)
        keys = ["Sigma"]
        assert all([k in list(args.keys()) for k in keys])
        assert all([str(x) != "None" for x in keys])
        assert 0 <= i and i < np.shape(self.X)[1]
        assert all([x == np.shape(self.X)[0] for x in np.shape(args["Sigma"])])
        assert t > 0
        v = lambda x : float(self.matrix_norm(x, args["Sigma"])*np.sqrt(2*self.beta(t)))
        ## individual
        if (str(j) == "None"):
            return v(self.X[:, i])
        ## paired
        else:
            assert 0 <= j and j < np.shape(self.X)[1]
            return v(self.X[:, i]-self.X[:, j])


    def index(self, i, j, t, args):
        gap_, var_ = self.gap(i, args, j=j), self.variance(i, t, args, j=j)
        return gap_+var_


    def calculate_B_ij(self, i, j, t,args=None):
        if (t != self.index_time):
            self.B_di = {}
            self.index_time = t
        b = self.B_di.setdefault((i,j), float(self.index(i,j,t, args=args)))
        return b



    def B_ij(self, i, j):
        return self.calculate_B_ij(i, j, 
                               self.t+int(self.T_init == 0), 
                               self.arg_index())
    
    def randf(self,x, f):
        #x_float = is_of_type_LIST(x, "float")
        #assert x_float or is_of_type_LIST(x, "int")
        t = (float if (type(x[0]==float)) else int)
        x = np.array(x, dtype=t)
        f_ = lambda z : t(f(z))
        #assert is_of_type(f_(x), "float" if (x_float) else "int")
        smp = sample(np.argwhere(x == f_(x)).flatten().tolist(), 1)[0]
        #assert is_of_type(smp, "int")
        return smp


    def m_maximal(self, x, m):
        '''Returns @m-sized list of maximizing elements in @x IN ORDER (ties broken randomly)'''
        #assert is_of_type_LIST(x, "float") or is_of_type_LIST(x, "int")
        #assert is_of_type(m, "int")
        K = len(x)
        assert m > 0 and m <= K
        ids = np.argsort(x).flatten().tolist()
        ids.reverse()
        if (m < K):
            ids = [None]*m
            for i in range(m):
                id_ = self.randf(x, np.max)
                ids[i] = id_
                x[id_] = -float("inf")
        return ids


    def m_max(self, values, m):
        '''Returns element in @values which is the @mth maximal element'''
        arg_values = np.argsort(np.array(values).flatten()).tolist()
        return values[arg_values[-m]]


    #' @param A Numpy matrix: inverse of matrix
    def iterative_inversion(self, A, x):
        return A - (np.dot(np.dot(A, x), np.dot(x.T, A)))/float(1+self.matrix_norm(x, A)**2)


    def arg_index(self):
        return {"Sigma": (self.sigma**2)*self.B_inv, "theta": self.theta}


    def compute_index(self, j):
        #index = float(np.max([self.B_ij(i, j) for i in self.notJ]))
        index = float(np.max([self.B_ij(i, j) for i in self.N_t]))
        return index


    def compute_nt(self):
        print("self.J",self.J, self.means)
        Jt_means = self.means[self.J]
        min_means = np.min(Jt_means)
        worst_index = sample(np.argwhere(self.means == min_means).flatten().tolist(), 1)[0]
        print("--- in compute nt ", worst_index)
        while worst_index not in self.J:
            worst_index = sample(np.argwhere(self.means == min_means).flatten().tolist(), 1)[0]
        assert worst_index in self.J
        return worst_index


    def select_arm(self,nt):
        Jt_means = self.means[self.J]
        bt_means = np.max(Jt_means)	
        st_means = self.means[self.challenger]
        ct_means = self.means[self.c_t]
        nt_means = self.means[nt]
        print("!!!!!!!!!!!!!!!!!!!!!!! challenger, bt_means, st_means", self.challenger, bt_means,self.means[self.best_arm], st_means)
        print("self.B_ij(self.challenger, self.best_arm)",self.B_ij(self.worst_arm,self.best_arm),self.B_ij(self.best_arm,self.worst_arm),"bt,st"+str(self.B_ij(self.best_arm,self.challenger)),"st,bt"+str(self.B_ij(self.challenger, self.best_arm)),"nt,challenger"+str(self.B_ij(self.worst_arm,self.challenger)),"challenger,nt"+str(self.B_ij(self.challenger,self.worst_arm)))
        #self.B_ij(nt,self.challenger)<=self.epsilon
        #self.B_ij(nt,self.challenger)<self.B_ij(self.challenger,self.best_arm)
        #(nt_means-st_means)<=self.epsilon
        if(ct_means > nt_means ):
            selected_arm = self.c_t
            assert selected_arm in self.N_t
            print("**********inside select_arm",self.N_t)
            if type(self.N_t)==list:
                self.N_t.remove(selected_arm)
            else:
                self.N_t = self.N_t.tolist()
                self.N_t.remove(selected_arm)
            print("notJ",self.notJ)
            #print("**********inside select_arm after removing",self.N_t)
            return selected_arm
        else:
            return None


    def compute_Jt(self):
        print("----------", self.t)
        if(self.t==0):
            print("Simulation", self.means)
            return self.m_maximal(self.means.tolist(), self.m)
        else:
            print("***************** TIME T =", self.t)
            nt = self.worst_arm
            selected_arm = self.select_arm(nt)
            print("-------- Before updating Jt", self.J)
            self.previous_J=self.J
            print("****** worst arm, selected arm", nt, selected_arm)
            if (selected_arm is not None):
                self.J.remove(nt)
                self.J.append(selected_arm)
            if type(self.N_t)==list:
                self.N_t.append(nt)
            else:
                self.N_t = self.N_t.tolist()
                self.N_t.append(nt)
            print("After removing and appending", self.J)
            J_means = self.means[self.J]
            inds = J_means.argsort()[::-1].tolist()
            print(inds)
            self.J = np.array(self.J)[inds].tolist()
            if self.J==self.previous_J:
                self.patience+=1
            else:
                self.patience=0
            print("After sorting",self.J)
            #J_means.sort(reverse=True)
            return self.J


    def compute_ct(self):
        #random.seed(42)
        #return self.J[randf(self.means[self.J], np.max)]
        Nt_means = self.means[self.N_t]
        max_means = np.max(Nt_means)
        best_index = sample(np.argwhere(self.means == max_means).flatten().tolist(), 1)[0]
        print("--- in compute bt ", best_index)
        while best_index not in self.N_t:
            best_index = sample(np.argwhere(self.means == max_means).flatten().tolist(), 1)[0]
        assert best_index in self.N_t
        return best_index


    def compute_bt(self):
        indices_J = [self.compute_index(i) for i in self.J]
        b_t = self.J[self.randf(indices_J, np.max)]
        return b_t


    def pull_arm_greedy(self, A):
        direction = self.X[:, self.best_arm]-self.X[:, self.challenger]
        uncertainty = [float(self.matrix_norm(direction, self.iterative_inversion(A, self.X[:,i]))) for i in self.arms]
        print("uncertainty",uncertainty)
        a = self.arms[self.randf(uncertainty, np.min)]
        print("arm a", a, self.best_arm, self.challenger)
        return [int(a)]


    def sampling_rule(self):
        self.is_greedy = True
        return (self.pull_arm_greedy if (self.is_greedy) else self.pull_arm_optimized)(self.B_inv)


    def sample(self):
        print("self.B_inv",self.B_inv.shape)
        self.J = self.compute_Jt()
        print("----- J, arms", self.J, self.arms)
        self.notJ = [a for a in self.arms if (a not in self.J)]
        if(self.t== 0):
            self.N_t = np.random.choice(np.array(self.notJ), self.n_das, replace=False)
        else:
            top_guys = np.sort(self.means[self.notJ])[::-1][:self.n_das].tolist()
            Q_t =  np.random.choice(np.array(self.notJ), self.n_das, replace=False)
            self.N_t = list(set(self.N_t).union(Q_t))
            self.N_t = [arm for arm in self.N_t if arm not in self.J]
            print("self.N_t",self.N_t)
            self.best_arm = self.compute_bt()
            indices_bt = [self.B_ij(a, self.best_arm) for a in self.N_t]
            top_n_das = np.sort(self.means[self.N_t])[::-1][:self.n_das] 
            print("top_n_das",top_n_das)
            self.N_t = [sample(np.argwhere(self.means == max_means).flatten().tolist(), 1)[0] for max_means in top_n_das] 
            print("self.N_t",self.N_t)
        nt = self.compute_nt()
        self.worst_arm = nt
        self.best_arm = self.compute_bt()
        indices_bt = [self.B_ij(a, self.best_arm) for a in self.N_t]
        self.challenger = self.N_t[self.randf(indices_bt, np.max)]
        self.c_t = self.compute_ct()
        print("J_t means", self.means[self.J])
        print("self.B_ij(self.challenger, self.best_arm)",self.B_ij(self.best_arm,self.worst_arm),"bt,st"+str(self.B_ij(self.best_arm,self.challenger)),"st,bt"+str(self.B_ij(self.challenger, self.best_arm)),"nt,challenger"+str(self.B_ij(self.worst_arm,self.challenger)),"challenger,nt"+str(self.B_ij(self.challenger,self.worst_arm)))
        return self.sampling_rule()


    def reward(self, score):
        return np.random.normal(score, self.sigma)


    def get_reward(self, arm):
        ## File IO error if several threads use the same file
        if (False):
            with open(self.scores_filepath, "r") as readFile:
                reader = csv.reader(readFile, delimiter=',')
                for row, record in enumerate(reader):
                    if (row == arm):
                        break
            try:
                record
            except:
                print((self.scores_filepath))
                print((os.path.exists(self.scores_filepath)))
                print("Resource is already used by another thread")
                raise ValueError
        else:
            record = [self.oracle[arm]]
        return self.reward(float(record[0]))


    def play(self, arm):
        assert arm in self.arms
        #print(self.subsets,self.subsets[14].shape,arm)
        reward = LLM_error_indicator(self.subsets[arm],self.val_data)
        self.rewards.append(reward)
        self.pulled_arms.append(arm)


    def learn_from_arms(self, candidates):
        print("``````````Learn from arms")
        if (len(candidates) > 0):
            if (len(candidates) > 0):
                for arm in candidates:
                    print("Inside for loop",candidates)
                    self.play(arm)
                    print("~~~~~~~~~~~~~~~~~~~ SELF.T",self.t)
                    self.t += 1
                    self.na[arm] += 1
                print("candidates",candidates,self.rewards,self.rewards[-1])
                self.update(candidates)



    def initialization(self, T_init=None):
        print("initialized")
        if (str(T_init) == None):
            T_init = self.T_init
        candidates = [[arm]*T_init for arm in self.arms]
        candidates = [y for x in candidates for y in x]
        assert len(candidates) == T_init*self.K
        self.learn_from_arms(candidates)


    def recommend_rule(self):
    	idx = [int(id_) for id_ in self.m_maximal(self.means.tolist(), self.m)]
    	means = self.means[idx].tolist()
    	assert all([m >= self.m_max(self.means.tolist(), self.m) for m in means])
    	return idx


    def recommend(self):
    	idx = self.recommend_rule()
    	idx = list(map(int, idx))
    	means = self.means[idx].tolist()
    	assert all([i in self.arms for i in idx]) and len(idx) == self.m
    	assert len(means) == self.m
    	return idx, means


    def update_linear_means(self, candidates):
        for i in range(1, (self.K*self.T_init if (self.T_init > 0 and self.t <= self.K*self.T_init) else len(candidates))+1):
            x = self.X[:, self.pulled_arms[-i]]
            self.B_inv = self.iterative_inversion(self.B_inv, x)
            self.b += self.rewards[-i]*np.array(x.tolist())[:,0]
            self.theta = np.dot(self.B_inv, self.b).reshape((1, self.N))
            self.means = np.array(np.dot(self.theta, self.X).tolist()[0])
        #print("alpha training error*****************",sum(self.rewards-self.means))
        print("-----***------------updated theta-------------------theta,means",self.theta,self.means)


    def update(self, candidates):
        print("-----------------Going to update theta-------------------")
        self.update_linear_means(candidates)


    def tau(self):
        st_bt = float(self.B_ij(self.challenger, self.best_arm))
        st_nt = float(self.B_ij(self.challenger, self.worst_arm))
        #st_nt = float(self.means[self.challenger] - self.means[self.worst_arm])
        return st_bt, st_nt


    def stopping_rule(self):
        self.B_st_bt, self.B_st_nt = self.tau()
        print("self.B <= self.epsilon",self.B_st_bt,self.B_st_nt,(self.B_st_bt <= self.epsilon) and (self.B_st_nt <= self.epsilon))
        return ((round(self.B_st_bt,2) <= self.epsilon))





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#-------------------------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def static_subset_selection(val_data, train_data, k, test_data):

    # test_data = test_data[:50]
    # test_data = 15, val_data=9, train_data=30-9=21, k=5, L=40, U=10, V=10, L-U=30
    # test_emb = torch.Tensor(test_emb)

    with open('../data/GSM8K/gsm8k_train_emb.pkl', 'rb') as f:
        train_emb = pickle.load(f)
    
    with open('../data/GSM8K/gsm8k_val_emb.pkl', 'rb') as f1:
        val_emb = pickle.load(f1)
    
    val_data = val_data[:20]
    val_emb = val_emb[:20]
    
    val_emb = torch.Tensor(val_emb)
    train_emb = torch.Tensor(train_emb)

    print("*****Embeddings loaded*****")
    # k-means clustering on train_data with k=5
    kmeans = KMeans(n_clusters=5, random_state=0).fit(train_emb)
    
    # Make clusters of train_data with same cluster label
    train_data['cluster'] = kmeans.labels_

    # # Do stratified sampling from train_data based on the first word of the question
    # train_data['w1'] = train_data['question'].str.split().str[0]

    # Create index column in train_data using arange
    train_data['index'] = np.arange(len(train_data))

    # Create L=_ subsets of size k total, with each group having k/num_gr examples
    num_gr = len(train_data['cluster'].unique())
    L = []
    L_indices = []

    # print("*****Initializing L*****")
    # Initialize L, 100 random set of subsets from train_data 
    for idx1 in range(100):
        subset = []
        for name, group in train_data.groupby('cluster'):
            # subset.append(group.sample(k//num_gr))   
            subset.append(group.sample(1))   
        subsets = pd.concat(subset)
        L.append(subsets)
        L_indices.append(subsets['index'].tolist())



    E_val = cosine_similarity(train_emb, val_emb)



    # ######### Making feature matrix of dim = (num_of_arms, num_of_val_data).T
    # len_L = len(L)
    # L_val = np.zeros((len_L,len(val_emb)))
    # for u in range(len_L):
    #     avg = np.zeros((1, len(val_emb)))
    #     for j in L_indices[u]:
    #         avg+= E_val[j]
    #     avg = avg/len(L_indices[u])
    #     L_val[u] = avg

    # N = val_data.shape[0]
    # K = len(L)
    # X = np.matrix(L_val.T) #np.matrix(np.eye(N,K))#np.random.normal(0, 0.05, size=(N, K))
    # #print("X",X,X.shape)  



    # ########### Making feature matrix of dim = (num_of_arms, num_of_ices).T
    len_L = len(L)
    L_val = np.zeros((len_L,len(L_indices[0])))
    for u in range(len_L):
        avg = []
        for j in L_indices[u]:
            avg.append(np.mean(E_val[j]))
        #avg = avg/len(L_indices[u])
        avg = np.array(avg).reshape(1,-1)
        L_val[u] = avg

    N = len(L_indices[0])# val_data.shape[0]
    K = len(L)
    X = np.matrix(L_val.T) #np.matrix(np.eye(N,K))#np.random.normal(0, 0.05, size=(N, K))
    #print("X",X,X.shape)   




    top_m_instance = top_m_arm(args={},X=X,
                               m=10,
                               val_data=val_data,
                               subsets=L)
    top_m_instance.clear()
    #top_m_instance.initialization(T_init=top_m_instance.T_init)


    i=0
    done=False
    round_mse = []
    acc_dict={}
    true_means = []
    arm_list = []
    top_m_instance.patience = 0

    while not done or top_m_instance.patience<20:
        print(("** Round #"+str(i+1)), end=' ')
        i += 1
        candidates = top_m_instance.sample()
        top_m_instance.learn_from_arms(candidates)
        done = top_m_instance.stopping_rule()

        if(i == 1):
            print("Will be calculating accuracy for all arms in J (val_data * J)llm calls")
            #true_means = [LLM_error_indicator(L[arm_i], val_data) for arm_i in top_m_instance.J]
            for arm_i in top_m_instance.J:
                if arm_i not in acc_dict:
                    acc_dict[arm_i] = LLM_error_indicator(L[arm_i], val_data)
                true_means.append(acc_dict[arm_i])
                arm_list.append(arm_i)
                
        else:
            if top_m_instance.c_t not in acc_dict:
                acc_dict[top_m_instance.c_t] = LLM_error_indicator(L[top_m_instance.c_t], val_data)

            # need to get proper worst index of J
            true_means.pop(top_m_instance.J.index(top_m_instance.worst_arm))
            arm_list.pop(top_m_instance.J.index(top_m_instance.worst_arm))

            # it is fine but J_t_means won't be having the last index as ct_mean as we are sorting J after appending ct
            # true_means.append(LLM_error_indicator(L[top_m_instance.c_t], val_data))

            true_means.append(acc_dict[top_m_instance.c_t])
            arm_list.append(top_m_instance.c_t)

        #J_t_means = [top_m_instance.means[arm_1] for arm_1 in top_m_instance.J]
        J_t_means = [top_m_instance.means[arm_1] for arm_1 in arm_list]
        #true_means = [true_means[arm_1] for arm_1 in top_m_instance.J]

        print("\n*************Approximation error of Validation Data on J ************")
        mse = ((np.array(J_t_means) - np.array(true_means))**2)
        round_mse.append(mse)
        print("\n mse between accuracies ", mse) 

    result = top_m_instance.recommend()
    print("result*******",result)
    best_arms = result[0]
    best_arms_subsets = [L[k] for k in best_arms]


    #===============================================================================
    # Calculate the pairwise overlap between the subsets in U
    overlaps=[]
    for i in range(len(best_arms_subsets)):
        inner_overlaps=[]
        for j in range(len(best_arms_subsets)):
            if i!=j:
                overlap=0
                for index_j, s_i in best_arms_subsets[i].iterrows():
                    for index_j, s_j in best_arms_subsets[j].iterrows():
                        if s_i["question"].lower() in s_j["question"].lower() or s_j["question"].lower() in s_i["question"].lower():
                            overlap+=1
                inner_overlaps.append(overlap)
        overlaps.append(inner_overlaps)
            
    print("\nOverlaps:",overlaps)
    print("Len of overlaps:",len(overlaps))

    overlap_for_subset = []
    avg_overlap = []
    min_overlap = []
    max_overlap = []
    overlap_for_each_subset = np.average(overlaps, axis=1)
    overlap_avg = np.average(overlap_for_each_subset)
    overlap_min = np.min(overlap_for_each_subset)
    overlap_max = np.max(overlap_for_each_subset)
    overlap_for_subset.append(overlap_for_each_subset.tolist())
    avg_overlap.append(overlap_avg.tolist()) 
    min_overlap.append(overlap_min.tolist())
    max_overlap.append(overlap_max.tolist())
    print("\n********* PAIRWISE OVERLAP *********")
    print("\nOverlap_for_subset:",overlap_for_subset)
    print("\nAVG_overlap:",avg_overlap)
    print("MIN_overlap:",min_overlap)
    print("MAX_overlap:",max_overlap)
    #===============================================================================

    folder1 = f"./output/loss_folder"
    np.savez(f'{folder1}', overlap_for_subset = overlap_for_subset , avg_overlap = avg_overlap, min_overlap = min_overlap, max_overlap = max_overlap, round_mse = round_mse)
    
    return best_arms_subsets
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





def get_open_source_completions(test_data, data):

    # stop_signal = "\n\n"
    matches = 0
    mismatches = 0

    question_df = {"question":[],"answers":[]}
    
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)

    print("*************Starting static subset selection*************")

    val_data = val_data[:20]

    exemplars = static_subset_selection(val_data, train_data, 5, test_data)

    print("*************Finished static subset selection*************")

    merged_exemplars = pd.concat(exemplars)
    merged_exemplars.to_csv("static_subset_selection_llama2_case.csv")

    # Alternatively you can just laod selected exemplars found in output folder
    #merged_exemplars = pd.read_csv("static_subset_selection_llama2_case.csv")
    #exemplars = [merged_exemplars[0:5],merged_exemplars[5:10],merged_exemplars[10:15],merged_exemplars[15:20],merged_exemplars[20:25],merged_exemplars[25:30],merged_exemplars[30:35],merged_exemplars[35:40],merged_exemplars[40:45],merged_exemplars[45:50]]

    print("\n\n\n********************Take the exemplar with minimum validation loss and use it as the exemplar")

    avg_err = LLM_avg_error(exemplars, val_data)
    print("\n\nAvg Error:",avg_err)
    ind = np.argmin(avg_err)
    print("\n\nMin index:",ind)
    exemplars = exemplars[ind]

    index=0
    acc_records = []
    exnum = 1

    for row in tqdm(test_data,total=len(test_data),desc="Generating"):

        prompt = prompt_for_manual_prediction(row, exemplars)
        tmp = llm_output(prompt)
        #print(len(tmp_list))
        #n = self_con(tmp_list)
        if len(tmp.split("Final Answer:"))>0:
            ans = tmp.split("Final Answer:")[-1]
            ans = ans.split("\n")[0]
            # print(ans)
            if "each" in ans:  ans = ans.split("each")[0]
            if "=" in ans: ans = ans.split("=")[-1]
            ans = re.sub(r'[^0-9.]',"",ans)
            if len(ans)>0 and ans[-1]==".": ans = ans[:-1]
            # print(ans, "**************")
            try:
                float(ans)
                ans = round(float(ans))
            except: pass
        
        ground_truth = float(clean_ans(row["answer"]))
        answer = ans
        if answer==ground_truth:
            matches+=1
        else: 
            mismatches += 1
        
        print("\nAnswer:", answer)
        print("Ground Truth:", ground_truth)

        question_df['question'].append(row["question"])
        question_df["answers"].append(answer)
        final_questions = pd.DataFrame(question_df)
        final_questions.to_csv("case_llama2_gsm8k.tsv",sep="\t",index=False)

        print("Accuracy:", matches/exnum)
        exnum += 1

    print("EM:", matches/(matches+mismatches))

    return final_questions

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_few_shot_prediction():

    ### Load data
    # with open("../data/GSM8K/gsm8k_train.jsonl", 'r') as f:
    #     json_list = list(f)
    # train_set = [json.loads(x) for x in json_list]

    train_set = pd.read_csv("../data/GSM8K/gsm8k_train.csv")

    with open("../data/GSM8K/gsm8k_test.jsonl", 'r') as f:
        json_list = list(f)
    test_set = [json.loads(x) for x in json_list]

    final_df = get_open_source_completions(test_set, train_set)
    print(final_df)


if __name__=='__main__':
    test_few_shot_prediction()
