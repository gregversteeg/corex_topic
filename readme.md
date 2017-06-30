# Anchored CorEx: Hierarchical Topic Modeling with Minimal Domain Knowledge

## Overview

The principle of *Cor*-relation *Ex*-planation has recently been introduced as a way to build rich representations that
are maximally informative about the data. This project optimizes the CorEx framework for sparse binary data, so that it can be leveraged for topic modeling. Our work demonstrates CorEx finds coherent, meaningful topics that are competitive with LDA topics across a variety of metrics, despite only utilizing binary counts.

This code also introduces an anchoring mechanism for integrating the CorEx topic model with domain knowledge via the information bottleneck. This anchoring is flexible and allows the user to anchor multiple words to one topic, one word to multiple topics, or any other creative combination in order to uncover topics that do not naturally emerge.

Detailed analysis and applications of the CorEx topic model using this code:<br>
[*Anchored Correlation Explanation: Topic Modeling with Minimal Domain Knowledge*](https://arxiv.org/abs/1611.10277), preprint 2017. [[bibtex]](https://scholar.googleusercontent.com/scholar.bib?q=info:coUlDN9XweQJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAWVaffALVl60ray1Op0cqZkOZPA1b_ADU&scisf=4&ct=citation&cd=-1&hl=en)

Underlying motivation and theory of CorEx:<br>
[*Discovering Structure in High-Dimensional Data Through Correlation Explanation*](http://arxiv.org/abs/1406.1222),
NIPS 2014.  [[bibtex]](https://scholar.googleusercontent.com/scholar.bib?q=info:92j_xtrqX_oJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAWVafobpFO4ed6EeXEbMQunUxHDHeuDgX&scisf=4&ct=citation&cd=-1&hl=en) <br>
[*Maximally Informative Hierarchical Representions of High-Dimensional Data*](http://arxiv.org/abs/1410.7404), 
AISTATS 2015. [[bibtex]](https://scholar.googleusercontent.com/scholar.bib?q=info:ZqTZyQdqI_UJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAWVaf3RJ7IOmG802hw7ZBnQ333f4mFDHj&scisf=4&ct=citation&cd=-1&hl=en)

This code can be used for any sparse binary dataset. In principle, continuous values in the range zero to one can also be used as 
inputs but the effect of this is not well tested. 

### Install

To install, download using [this link](https://github.com/gregversteeg/corex_topic/archive/master.zip) 
or clone the project by executing this command in your target directory:
```
git clone https://github.com/gregversteeg/corex_topic.git
```
Use *git pull* to get updates. The code is under development. 
Please contact me about issues with this pre-alpha version.  

### Dependencies

CorEx requires numpy and scipy. If you use OS X, I recommend installing the [Scipy Superpack](http://fonnesbeck.github.io/ScipySuperpack/).

The visualization capabilities in vis_topic.py require other packages: 
* matplotlib - Already in scipy superpack.
* [networkx](http://networkx.github.io)  - A network manipulation library. 
* sklearn - Already in scipy superpack and only required for visualizations. 
* [graphviz](http://www.graphviz.org) (Optional, for compiling produced .dot files into pretty graphs. The command line 
tools are called from vis_topic. Graphviz should be compiled with the triangulation library for best visual results).

## Usage

### Command Line

```python
python vis_topic.py tests/data/twenty.txt --n_words=2000 --layers=50,5,1 -v --edges=150 -o test_output
```

### Python API

Given a doc-word matrix, the CorEx topic model is easy to train. The code follows the scikit-learn fit/transform conventions.

```python
import corex_topic as ct
import vis_topic as vt
import scipy.sparse as ss

# Define a matrix where rows are samples (docs) and columns are features (words)
X = np.array([[0,0,0,1,1],
              [1,1,1,0,0],
              [1,1,1,1,1]], dtype=int)
# Sparse matrices are also supported 
X = ss.csr_matrix(X)

# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=2)  # Define the number of latent topcis to use.
topic_model.fit(X)
```

To run twenty newsgroups, you can first run /tests/data/get_twenty.py to get a sparse matrix and then load and run it
within ipython or whatever. 

### CorEx outputs

```python
layer1.clusters  # Each variable/column is associated with one Y_j
# array([0, 0, 0, 1, 1])
layer1.labels[:, 0]  # Labels for each sample for Y_0
# array([0, 0, 1, 1])
layer1.labels[:, 1]  # Labels for each sample for Y_1
# array([0, 1, 0, 1])
layer1.tcs  # TC(X;Y_j) (all info measures reported in nats). 
# array([ 1.385,  0.692])
# TC(X_Gj) >=TC(X_Gj ; Y_j)
# For this example, TC(X1,X2,X3)=1.386, TC(X4,X5) = 0.693

# If you ran python get_twenty.py then 
twenty = cPickle.load(open('tests/data/twenty_mat20000.dat'))
out = ct.Corex(n_hidden=100, verbose=True, max_iter=100).fit(twenty)
words = cPickle.load(open('tests/data/dictionary20000.dat'))
vt.vis_rep(out, column_label=words)
```

As shown in the example, *clusters* gives the variable clusters for each hidden factor Y_j and 
*labels* gives the labels for each sample for each Y_j. 
Probabilistic labels can be accessed with *p_y_given_x*. 

The total correlation explained by each hidden factor, TC(X;Y_j), is accessed with *tcs*. Outputs are sorted
so that Y_0 is always the component that explains the highest TC. 
Like point-wise mutual information, you can define point-wise total correlation measure for an individual sample, x^l     
TC(X = x^l;Y_j) == log Z_j(x)   
This quantity is accessed with *log_z*. This represents the correlations explained by Y_j for an individual sample.
A low (or even negative!) number can be obtained. This can be interpreted as a measure of how surprising an individual
observation is. This can be useful for anomaly detection. 

See the main section of vis_topic.py for more ideas of how to do visualization.

### Generalizations

#### Hierarchical CorEx
The simplest extension is to stack CorEx representations on top of each other. 
```
layer1 = ct.Corex(n_hidden=100)
layer2 = ct.Corex(n_hidden=10)
layer3 = ct.Corex(n_hidden=1)
Y1 = layer1.fit_transform(X)
Y2 = layer2.fit_transform(np.matrix(Y1.labels))
Y3 = layer2.fit_transform(np.matrix(Y2.labels))
```
The sum of total correlations explained by each layer provides a successively tighter lower bound on TC(X) (see AISTATS paper). 
 To assess how large your representations should be, look at quantities
like layer.tcs. Do all the Y_j's explain some correlation (i.e., all the TCs are significantly larger than 0)? If not
you should probably use a smaller representation.

#### Getting better results

Note that CorEx can find different local optima after different random restarts. You can run it k times and take
the best solution (with the highest TC).
This version only allows for tree structure (a word can be in only one topic). In principle, we could use newer results
from the AISTATS paper to get around this but it slows things down a lot. 

#### Better visualizations
If you install the gts library and THEN install graphviz, graphviz should be capable of better visualizations (by using
gts to render nicer curved lines). 
```
sfdp tree.dot -Tpdf -Earrowhead=none -Nfontsize=12  -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True -Gpackmode=clust -Gsep=0.02 -Gratio=0.7 -Gsplines=True -o nice.pdf
```

## Technical notes

For speed this version actually works only on binary data and produces binary latent factors. However, you can 
interpret input values in the range 0 to 1 as specified a probability that a word appears in a document. Therefore,
we have several strategies for handling text data. 

At the command line API you can specify the following strategies for handling non-binary count data.
 
 0. Naive binarization. This will be good for documents of similar length and especially short documents. 
 
 1. Average binary bag of words. We split
                        documents into chunks, compute the binary bag of words
                        for each documents and then average. This implicitly
                        weights all documents equally. 
                        
 2. All binary bag of words. Split documents into chunks and consider each chunk as its own binary bag of words documents. 
 This changes the number of documents so it may take some work to match the ids back, if desired. Implicitly, this
 will weight longer documents more heavily. Generally this seems
 like the most theoretically justified method to me. Ideally, you could aggregate the latent factors over sub-documents to
 get 'counts' of latent factors at the higher layers. 
 
 3. Fractional counts. This converts counts into a fraction of the
                        background rate, with 1 as the max. Short documents
                        tend to stay binary and words in long documents are
                        weighted according to their frequency with respect to
                        background in the corpus. This seems to work Ok on tests. It requires no preprocessing of count 
                        data and it uses the full range of possible inputs. But it's not a very rigorous approach.
                        
 For the python API, for 1 and 2, you can use the functions in vis_topic.py to process data or do the same yourself.
 0 is specified through the python api with count='binarize' and 3 with count='fraction'. 



## Licensing
This version is free to use for academic and non-commercial purposes. For commercial uses, this code is free to try 
for 30 days. Please contact us for information on licensing arrangements. 

## Example results
This was obtained using the 20 newsgroups dataset with binarized bag of words. We first split up documents to
have a maximum length of 300 words. We used a snowball stemmer. The data was downloaded using sklearn with 
headers footers and comments (supposedly) removed. 
Hierarchical structure shown below.

0:q,m,z,v,ax,p,w,max,pl,bhj
1:file,window,program,softwar,ftp,user,server,version,dos,graphic
2:game,team,player,play,season,hockey,leagu,score,playoff,basebal
3:r,n,f,l,h,g,b,k,d,y
4:card,disk,pc,drive,mac,ram,machin,scsi,video,driver
5:armenian,turkish,armenia,turk,azerbaijani,turkey,azerbaijan,ottoman,sumgait,genocid
6:montreal,calgari,chicago,smith,blue,quebec,king,minnesota,patrick,loui
7:jew,israel,war,arab,muslim,isra,jewish,palestinian,nazi,villag
8:god,his,christian,he,him,jesus,christ,bibl,church,religion
9:govern,law,gun,crime,enforc,countri,crimin,polit,weapon,citizen
10:medic,diseas,patient,doctor,infect,treatment,clinic,medicin,hicnet,newslett
11:orbit,space,launch,solar,earth,shuttl,satellit,design,moon,circuit
12:that,it,of,to,a,the,difficult,repeat,miss,btw
13:do,if,you,have,your,my,know,want,how,i
14:who,children,live,kill,her,she,death,against,women,men
15:sin,lord,scriptur,paul,passag,spirit,matthew,biblic,john,gospel
16:x,contrib,lcs,tar,xt,de,lib,fr,uu,det
17:car,bike,ride,mile,brake,wheel,rear,motorcycl,dod,honda
18:encrypt,key,clipper,secur,escrow,privaci,chip,nsa,algorithm,des
19:was,were,had,they,did,said,say,didnt,believ,went
20:research,univers,publish,studi,book,scienc,institut,organ,scientif,author
21:nation,presid,russian,offic,unit,militari,administr,agenc,soviet,fund
22:jpeg,ray,map,astronom,manipul,telescop,rayshad,arc,gamma,jfif
23:nasa,mission,spacecraft,inc,contact,date,astronomi,mar,internet,ame
24:price,sale,sell,system,cost,buy,offer,ship,market,product
25:this,be,with,is,and,extrem,familiar,evalu,corrupt,trend
26:thank,pleas,mail,me,email,appreci,send,am,help,ani
27:gif,color,convert,zip,au,sgi,viewer,fax,tcp,wuarchiv
28:md,st,al,et,sj,nl,ai,ed,cl,ab
29:avail,anonym,mit,archiv,distribut,export,usenet,binari,output,host
30:human,evid,atheist,moral,teach,interpret,contradict,atheism,context,reject
31:san,april,los,center,california,angel,francisco,jose,divis,confer
32:edu,com,cs,ca,apr,gov,uiuc,gmt,ac,netcom
33:text,contain,set,function,messag,sourc,list,document,standard,command
34:we,fact,person,clear,statement,life,understand,matter,agre,action
35:homosexu,sexual,sex,male,gay,food,blood,eat,heterosexu,mari
36:at,are,other,in,two,least,both,abil,popular,comparison
37:gas,water,heat,tank,floor,tear,hot,air,ground,door
38:so,what,dont,then,im,here,sure,anyth,put,exact
39:would,becaus,thing,could,reason,realli,probabl,enough,actual,wrong
40:fbi,koresh,batf,fire,compound,waco,davidian,agent,atf,reno
41:or,use,can,for,need,look,either,etc,type,capabl
42:hous,citi,washington,york,senat,east,depart,street,white,school
43:provid,support,base,develop,number,requir,subject,specif,addit,detail
44:than,more,less,better,rather,much,wors,easier,smaller,bigger
45:out,when,go,been,good,still,turn,few,long,until
46:year,ago,month,money,week,old,pay,tax,spend,plan
47:time,no,one,well,now,over,take,sinc,come,while
48:their,those,own,day,made,dure,today,member,place,offici
49:mamma,shout,thou,saudi,arabia,kuwait,karina,lyuda,marina,igor
50:state,issu,continu,report,involv,press,attempt,form,activ,individu
51:all,some,onli,will,way,differ,howev,give,anoth,choos
52:by,from,must,sever,exampl,found,view,describ,thus,appear
53:effect,result,larg,level,limit,common,reduc,usual,amount,lower
54:includ,follow,note,new,current,section,full,receiv,process,below
55:even,right,after,again,left,never,though,hand,certain,onc
56:on,an,work,possibl,allow,direct,fine,similar,recent,basic
57:world,face,fight,decid,fall,decis,taken,conclud,minor,movement
58:line,phone,bit,servic,commerci,box,telephon,communic,technic,releas
59:three,high,increas,four,near,averag,five,cover,half,factor
60:get,like,up,problem,around,sound,abl,great,friend,unfortun
61:guy,night,hes,watch,oh,beat,stay,yeah,hey,flame
62:vice,bobb,versa,beauchain,ico,sank,bronx,manhattan,tek,queen
63:down,off,let,littl,side,sit,pull,alway,shoot,stick
64:first,second,between,begin,later,major,previous,entir,consist,comment
65:question,answer,which,there,without,simpl,necessari,sacrif,coher,disassembl
66:point,whi,whether,care,cours,feel,becom,expect,concern,realiz
67:as,mani,may,purpos,term,approach,cat,encompass,improp,meantim
68:too,back,happen,bad,mayb,couldnt,theyr,rememb,poor,chanc
69:argument,claim,true,fals,prove,therefor,mere,definit,convinc,behavior
70:but,same,find,chang,quit,suggest,correct,instead,troubl,strang
71:just,think,got,lot,everyth,seen,perfect,mess,silli,impress
72:our,us,neighbor,strong,held,trust,proof,approv,burden,gather
73:such,general,case,part,order,protect,prevent,initi,itself,appropri
74:stop,behind,serious,mind,lose,caught,hurt,push,straight,tough
75:mean,discuss,read,valid,knowledg,basi,logic,unless,presum,inher
76:not,doe,cannot,incorrect,fatal,liabl,sloppi,recognis,ascertain,enlighten
77:peopl,judg,rule,abus,respect,consequ,hide,assumpt,critic,qualifi
78:see,them,should,show,real,easili,dozen,counter,clockwis,discus
79:most,import,through,although,within,maintain,next,success,along,character
80:about,has,kind,deal,short,fill,constant,wherea,introduc,afterward
81:sens,lack,perhap,necessarili,nor,inde,pure,equal,distinct,bias
82:soon,bank,shame,nervous,sixti,anytim,stranger,breakdown,railroad,plaqu
83:light,eye,led,heavi,across,visibl,quiet,duti,deep,surround
84:accept,practic,act,accord,relationship,fundament,strict,recogn,emphasi,aris
85:word,exist,appli,simpli,confus,yourself,regard,essenc,conform,simplifi
86:oper,each,origin,present,special,higher,ident,precis,uniqu,variant
87:group,particular,given,ask,convers,creation,aspect,wrote,construct,conveni
88:third,period,field,lead,close,contribut,kept,candid,earn,theoret
89:into,these,where,formula,intrud,surpass,simplic,comedi,oposit
90:man,bear,stand,grant,amaz,broke,settl,ultim,disput,defeat
91:road,dark,black,border,mountain,room,scene,grey,silver,axe
92:consid,self,opinion,attitud,exercis,conserv,philosophi,criteria,correl,categori
93:non,complex,ad,extend,orient,equival,calcul,primarili,fulli,henc
94:last,morn,sunday,earli,regular,credit,relief,televis,span,dish
95:caus,occur,safe,danger,failur,excess,unusu,interfer,difficulti,stress
96:public,search,licens,intellig,speech,wish,permiss,acknowledg,artifici,identif
97:stuff,delet,hard,replac,soft,bar,master,parallel,doubl,simpler
98:also,veri,under,widespread,wealth,vocal,overhaul,reread,idealist
99:start,move,minut,break,forward,finger,count,split,match,noon

The colors are just added to draw attention to things I found interesting at first glance. 
![hierarchical topics](https://github.com/gregversteeg/corex_topic/blob/master/tests/data/color_topic.png?raw=true "hierarchical topics")
