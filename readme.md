# Hierarchical Topic Modeling of Sparse Count Data with CorEx

The principle of *Cor*-relation *Ex*-planation has recently been introduced as a way to build rich representations that
are maximally informative about the data. This project consists of python code to build these representations. 
The technique is generally more similar to the NIPS paper then the later AISTATS for speed reasons. This version
is optimized for sparse binary data. In principle, continuous values in the range zero to one can also be used as 
inputs but the effect of this is not well tested. 

A preliminary version of the technique is described in this paper.      
[*Discovering Structure in High-Dimensional Data Through Correlation Explanation*](http://arxiv.org/abs/1406.1222), 
NIPS 2014.

Further theoretical developments are described here:      
[*Maximally Informative Hierarchical Representions of High-Dimensional Data*](http://arxiv.org/abs/1410.7404), 
AISTATS 2015.  

###Dependencies

CorEx requires numpy and scipy. If you use OS X, I recommend installing the [Scipy Superpack](http://fonnesbeck.github.io/ScipySuperpack/).

The visualization capabilities in vis_topic.py require other packages: 
* matplotlib - Already in scipy superpack.
* [networkx](http://networkx.github.io)  - A network manipulation library. 
* sklearn - Already in scipy superpack and only required for visualizations. 
* [graphviz](http://www.graphviz.org) (Optional, for compiling produced .dot files into pretty graphs. The command line 
tools are called from vis_topic. Graphviz should be compiled with the triangulation library for best visual results).

###Install

To install, download using [this link](https://github.com/gregversteeg/corex_topic/archive/master.zip) 
or clone the project by executing this command in your target directory:
```
git clone https://github.com/gregversteeg/corex_topic.git
```
Use *git pull* to get updates. The code is under development. 
Please contact me about issues with this pre-alpha version.  

## Example usage with command line interface

This implementation is optimized for sparse count data. Here is an example using the command line interface.
```python
python vis_topic.py tests/data/twenty.txt --n_words=2000 --layers=20,3,1 -v --edges=50 -o test_output
```

## Python API usage

### Example

```python
import corex_topic as ct
import vis_topic as vt
import scipy.sparse ass

X = np.array([[0,0,0,0,0], # A matrix with rows as samples and columns as variables.
              [0,0,0,1,1],
              [1,1,1,0,0],
              [1,1,1,1,1]], dtype=int)
X = ss.csr_matrix(X)  # Sparse matrices are supported, or np.matrix(X) for dense. 

layer1 = ct.Corex(n_hidden=2)  # Define the number of hidden factors to use.
layer1.fit(X)

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

To run twenty newsgroups, you can first run /tests/data/get_twenty.py to get a sparse matrix and then load and run it
within ipython or whatever. 

### CorEx outputs

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

## Licensing
This version is free to use for academic and non-commercial purposes. For commercial uses, this code is free to try 
for 30 days. Please contact us for information on licensing arrangements. 

## Example results
This was obtained using the 20 newsgroups dataset with binarized bag of words. We first split up documents to
have a maximum length of 300 words. We used a snowball stemmer. The data was downloaded using sklearn with 
headers footers and comments (supposedly) removed. 
Hierarchical structure shown below.

0:z,m,q,v,n,l,p,w,r,f
1:window,file,softwar,program,card,dos,disk,pc,user,graphic
2:game,team,player,play,season,hockey,leagu,score,playoff,win
3:ftp,pub,x,server,mit,edu,anonym,directori,sun,archiv
4:but,have,that,this,it,if,would,some,is,like
5:kill,law,children,jew,war,crime,death,muslim,murder,israel
6:god,christian,jesus,bibl,christ,church,sin,faith,scriptur,lord
7:armenian,turkish,armenia,turk,azerbaijani,turkey,genocid,azerbaijan,massacr,russian
8:includ,avail,inform,research,develop,data,sourc,public,contain,publish
9:as,are,be,not,one,there,onli,in,of,and
10:do,dont,what,so,say,about,your,you,becaus,think
11:nation,april,citi,york,center,washington,san,los,north,east
12:who,peopl,those,govern,live,us,against,countri,world,american
13:price,high,engin,design,cost,sale,sell,power,low,electron
14:medic,diseas,patient,doctor,infect,treatment,medicin,health,clinic,geb
15:year,been,out,after,over,last,now,day,down,back
16:he,his,was,were,had,him,did,said,didnt,came
17:nasa,orbit,space,gov,mission,shuttl,satellit,earth,solar,spacecraft
18:with,use,or,can,ani,for,work,problem,need,help
19:up,when,on,get,go,my,off,veri,good,put
20:encrypt,key,clipper,secur,escrow,algorithm,nsa,privaci,chip,des
21:state,presid,feder,clinton,offic,congress,fund,hous,administr,press
22:human,fact,life,act,natur,statement,moral,attempt,polici,legal
23:believ,claim,reason,certain,argument,evid,belief,forc,truth,matter
24:launch,water,moon,rocket,atmospher,air,ground,fuel,lunar,nuclear
25:than,more,less,much,better,rather,far,higher,averag,easier
26:at,least,two,end,order,increas,home,plan,month,hour
27:other,which,these,may,result,effect,caus,either,non,found
28:word,point,exist,read,view,cannot,object,defin,sens,written
29:we,our,must,clear,understand,feel,true,ignor,obvious,love
30:right,again,keep,care,mind,serious,eye,wors,realiz,pain
31:well,too,though,might,lot,probabl,enough,mayb,pretti,yet
32:them,no,see,way,take,let,give,yes,hope,especi
33:time,has,then,make,sinc,while,long,here,real,goe
34:accept,therefor,nor,respons,upon,individu,among,neither,reject,interpret
35:first,second,three,third,final,period,later,cover,head,total
36:meet,bodi,news,famili,stand,five,earli,sign,lead,cmu
37:post,discuss,group,newsgroup,issu,alt,specif,topic,sci,reader
38:author,subject,refer,present,name,articl,form,origin,letter,intend
39:requir,each,complet,oper,continu,build,full,pass,local,extend
40:mean,case,whether,simpli,cours,explain,exact,rule,sort,impli
41:dure,close,mass,due,signific,factor,consider,success,impact,cloud
42:will,into,line,unless,easi,fit,break,otherwis,fail,piec
43:their,they,rare,fashion,audienc,struck,pacifist,applaud,trot,backbon
44:system,standard,process,servic,document,intern,various,add,track,correspond
45:abov,within,examin,togeth,separ,mention,appar,moment,resolv,led
46:such,becom,restrict,necessari,prove,propos,wish,logic,econom,search
47:through,between,where,direct,allow,complex,edg,seek,analog,reli
48:white,black,green,gold,edt,eric,mark,gore,mack,greg
49:own,often,major,deal,fair,busi,pay,youll,equival,along
50:an,similar,event,return,answer,except,everyth,evalu,expos,trail
51:from,under,prevent,reach,asid,excerpt,closest,refrain,cancel,wipe
52:involv,began,particip,whose,demonstr,train,throughout,invit,creation,david
53:call,support,find,write,alreadi,simpl,adopt,recogn,discov,flag
54:by,member,known,identifi,prior,charter,identif,plausibl,incomplet,phenomena
55:general,relat,describ,affect,frequent,vari,ordinari,popular,formal,accomplish
56:person,opinion,accord,grant,employ,fall,disclaim,alon,encourag,favor
57:also,provid,main,method,background,subsequ,divid,versa,robust,unreli
58:follow,note,open,improv,initi,focus,jennif,placement,quota,slipper
59:new,interest,industri,sub,colleg,zealand,deliv,soda,sixteen,visa
60:current,ad,capabl,typic,enhanc,measur,maximum,zero,bell,multipli
61:base,number,free,calcul,aim,marker,whilst,optimum,clyde,hind
62:purpos,creat,draw,independ,deriv,construct,associ,usag,core,permiss
63:itself,valu,common,content,ident,approach,properti,status,rapid,shelter
64:mani,show,almost,necessarili,bias,devot,oblig,inter,pbs,zillion
65:place,situat,posit,begin,thus,consist,enthusiast,lengthi,overlook,pertin
66:import,without,abil,although,hold,grow,skill,hidden,perman,realist
67:carri,communiti,outsid,parent,readi,sister,stone,aggress,contrast,soil
68:made,decid,parti,mistak,imagin,scenario,middl,amaz,defenc,casual
69:consid,risk,avoid,safeti,approv,step,strict,furthermor,entitl,intercept
70:all,chang,yellow,deep,restor,unsuccess,gday,frenzi,guarente,helluva
71:potenti,short,resist,easili,safe,balanc,ensur,characterist,attract,isol
72:part,exampl,educ,substitut,academ,exceed,facilit,privileg,repetit,shudder
73:remain,million,organ,instanc,assur,princip,twelv,remaind,ant,slant
74:activ,observ,indic,confirm,gather,environment,penetr,disrupt,appal,blur
75:most,learn,huge,henc,partial,assess,newbi,evad,crari,shrewd
76:given,equal,comment,tim,andi,superior,stood,competit,predict,mat
77:level,depend,essenti,random,overal,pattern,estim,minimum,satisfi,constrain
78:receiv,sent,ask,glare,winston,comedi,nad,morton,wrought,wand
79:particular,term,role,broad,conform,crucial,prone,regress,orgin,prop
80:both,knowledg,miss,photograph,unusu,exploit,fred,aerial,pearl,stanza
81:sever,section,numer,immedi,pictur,reconstruct,plaqu,mgr,retrofit,mole
82:quit,expect,rest,happi,matur,pace,entertain,dubious,hierarch,extol
83:anoth,instead,primarili,match,swear,orang,preclud,badg,firebomb,oberon
84:entir,percent,decad,exclus,ture,specimen,despair,guerrilla,unmistak,knx
85:appear,further,guarante,target,subset,torch,extraordinari,spectr,beaver,relic
86:perfect,choic,quick,fanci,blade,outfit,sheer,halfway,petti,similiar
87:leav,enter,cso,alert,poke,uxa,brush,deflect,immin,vet
88:eventu,greater,conduct,growth,tremend,likewis,span,waiver,clay,aval
89:accur,substanti,plane,circl,burst,circular,chamber,vanish,propag,inhibit
90:extrem,degre,distort,theoret,weigh,retain,compens,prolong,alic,dutch
91:previous,maintain,scene,temporari,fragment,unfamiliar,ponder,acheiv,geeki,eyebal
92:basic,precis,simplifi,mutat,gate,masterpiec,woo,laura,hvo,labl
93:fill,neutral,primit,underground,brigham,stain,provo,fiat,aqua,earmark
94:statist,liter,correl,minus,tenth,tricki,widen,taiwan,crafti
95:tendenc,argh,westcott,beza,bereft,vaticanus,epp,rightw,sanctiti,dashnagtzoutun
96:sight,stronger,twist,intrigu,amelior,hunk,wattl,characht,archo,horsemen
97:vic,pps,marbl,vco,iol,vcf,imi,qu,iy,rv
98:scream,meal,shirt,lancast,sweeptak,alder,calv,armenian,cancun,stubbl
99:brad,rumbl,archibald,wiemer,boschman,templeton,dreamer,lostweekend,bandwaggon,smeghead

![hierarchical topics](https://github.com/gregversteeg/corex_topic/blob/master/tests/data/topic_hierarchy.pdf?raw=true "hierarchical topics")