// Build editable defense slides from the final report and frozen source tables.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';

const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const modules=process.env.PR02_NODE_MODULES;
const skill=process.env.PR02_PRESENTATIONS_SKILL;
const python=process.env.PR02_RUNTIME_PYTHON;
if(!modules||!skill||!python)throw new Error('Set PR02_NODE_MODULES, PR02_PRESENTATIONS_SKILL and PR02_RUNTIME_PYTHON.');
process.env.RUNTIME_NODE_MODULES ||= modules;
const require=createRequire(path.join(modules,'entry.cjs'));
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const build=path.resolve(process.env.PR02_SLIDES_BUILD||path.join(repo,'work/defense_20261014'));
const finalPath=path.resolve(process.env.PR02_PPTX_OUTPUT||path.join(repo,'reports/defense/PR02-01_DEFENSE.pptx'));
await fs.mkdir(build,{recursive:true});await fs.mkdir(path.dirname(finalPath),{recursive:true});
const sources={};
async function read(name){const bytes=await fs.readFile(path.join(repo,name));sources[name]=crypto.createHash('sha256').update(bytes).digest('hex');return bytes.toString('utf8');}
function parse(text,sep=','){
 const result=[];let row=[],field='',quoted=false;
 for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}
 else if(c===sep&&!quoted){row.push(field);field='';}else if(c==='\n'&&!quoted){row.push(field);result.push(row);row=[];field='';}else if(c!=='\r')field+=c;}
 if(field||row.length){row.push(field);result.push(row);}const head=result.shift();return result.filter(r=>r.length===head.length).map(r=>Object.fromEntries(head.map((h,i)=>[h,r[i]])));
}
async function csv(name){return parse(await read(name),name.endsWith('.tsv')?'\t':',');}
const R0='runs/m4_complete_20261014_v1',R1='runs/m4_refinement_20261014_v2';
const report=JSON.parse(await read('reports/final/report_content.json'));
const old=Object.fromEntries((await csv(R0+'/evaluation/model_summary.csv')).map(r=>[r.method,r]));
const refined=Object.fromEntries((await csv(R1+'/regularization/summary.csv')).map(r=>[r.method,r]));
const spatial=Object.fromEntries((await csv(R1+'/grouped/summary.csv')).map(r=>[r.method,r]));
const oldStages=Object.fromEntries((await csv(R0+'/ablation/stage_summary.csv')).map(r=>[r.stage,r]));
const region=await csv(R1+'/attribution_stability/region_by_seed.csv');
const across=await csv(R1+'/attribution_stability/cross_seed_summary.csv');
const within=await csv(R0+'/explanation_analysis/consistency_summary.csv');
const mutation=await csv(R0+'/mutation/mutation_summary.csv');
const direction=(await csv(R0+'/mutation/direction_summary.csv')).find(r=>r.expected_direction==='all');
const transfer=await csv(R1+'/transfer_stability/summary.csv');
const data=await csv('data/03_Ecoli_strength87/samples.tsv');
const protocol=JSON.parse(await read('configs/m4_refinement_v2.json'));
const train=data.filter(r=>r.split==='train');
const p=Presentation.create({slideSize:{width:1280,height:720}});
const FONT='Source Han Sans CN';
const C={ink:'#172F3B',muted:'#5E727D',blue:'#285B70',teal:'#247F81',orange:'#C37C3A',gray:'#AAB8C1',pale:'#E9F0F3',grid:'#DDE5E9'};
const f=(v,n=4)=>Number(v).toFixed(n).replace(/^-0\.0000$/,'0.0000');
const pct=v=>(100*Number(v)).toFixed(2)+'%';
const r2=r=>Number(r.r2_log10_mean),rho=r=>Number(r.spearman_mean);
const titles=[],nativeCharts=[],nativeTables=[],evidence=[];
function text(s,value,x,y,w,h,size=27,bold=false,color=C.ink,align='left'){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=String(value);t.text.style={typeface:FONT,fontSize:size,bold,color,alignment:align,verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0,lineSpacing:1.15};return t;
}
function slide(title,context,sourcesForSlide=[],explanation=''){
 const s=p.slides.add();s.background.fill='#FFFFFF';titles.push(title);
 text(s,title,64,38,1152,70,43,true);
 if(context)text(s,context,66,116,1146,43,23,false,C.muted);
 text(s,String(p.slides.items.length).padStart(2,'0'),1174,676,42,26,17,false,C.muted,'right');
 s.speakerNotes.text=[explanation,'数据与方法依据：',...sourcesForSlide].join('\n');return s;
}
function table(s,headers,rows,{x=66,y=188,w=1148,h=360,widths,size=24,highlight=[],padding=7}={}){
 const t=s.tables.add({rows:rows.length+1,columns:headers.length,left:x,top:y,width:w,height:h,values:[headers,...rows],columnWidths:widths});
 t.borders.assign({fill:C.grid,width:.7,style:'solid'});
 for(let i=0;i<=rows.length;i++)for(let j=0;j<headers.length;j++){
  const cell=t.getCell(i,j);cell.fill=i===0?C.pale:'#FFFFFF';
  cell.text.style={typeface:FONT,fontSize:size,bold:i===0||highlight.includes(i-1),color:highlight.includes(i-1)?C.teal:C.ink,
   alignment:j===0?'left':'center',verticalAlignment:'middle',insets:{left:12,right:12,top:padding,bottom:padding}};
 }
 nativeTables.push(p.slides.items.length);evidence.push({slide:p.slides.items.length,kind:'table',values:[headers,...rows]});return t;
}
function chart(s,categories,series,{x=66,y=185,w=760,h=400,ymin,ymax,major,format='0.00',legend=true,labels=false,type='bar',ylabel='',labelsize=23,gap=90}={}){
 const normalized=series.map((r,i)=>({...r,values:r.values.map(v=>v===null?null:Number(Number(v).toFixed(10))),fill:r.fill||[C.blue,C.teal,C.orange,C.gray][i%4],line:r.line||{fill:r.fill||[C.blue,C.teal,C.orange,C.gray][i%4],width:3},marker:type==='line'?{symbol:'circle',size:7}:undefined}));
 const c=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},categories,series:normalized,
  hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{fontSize:22}},
  barOptions:{direction:'column',grouping:'clustered',gapWidth:gap},lineOptions:{grouping:'standard',smooth:false},
  xAxis:{textStyle:{fontSize:labelsize},line:{fill:C.gray,width:1},tickLabelPosition:'low'},
  yAxis:{title:ylabel?{text:ylabel,textStyle:{fontSize:22}}:undefined,numberFormatCode:format,min:ymin,max:ymax,majorUnit:major,textStyle:{fontSize:21},majorGridlines:{fill:C.grid,width:.7},line:{fill:'none',width:0}},
  dataLabels:labels?{showValue:true,position:'outEnd',numberFormatCode:format.includes('%')?format:'0.000',textStyle:{fontSize:22,color:C.ink}}:undefined,
  chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}});
 applyPresentationChartFont(c,{fontFamily:FONT});nativeCharts.push(p.slides.items.length);
 evidence.push({slide:p.slides.items.length,kind:'chart',categories,series:normalized.map(r=>({name:r.name,values:r.values}))});return c;
}
function finding(s,value,detail,{x=896,y=210,w=306,color=C.teal}={}){
 text(s,value,x,y,w,75,44,true,color);text(s,detail,x,y+90,w,160,27);
}

// 01
{
 const s=p.slides.add();s.background.fill='#F6F9FA';titles.push('启动子强度预测与关键位点归因');
 text(s,'启动子强度预测\n与关键位点归因',78,156,1120,190,64,true);
 text(s,'PR02-01 项目答辩',82,386,1080,58,32,false,C.teal);
 text(s,report.members.join('  '),82,511,1120,48,26);
 text(s,'2026年10月14日',82,583,1000,38,23,false,C.muted);
 s.speakerNotes.text='本研究比较大肠杆菌启动子相关表达的预测模型，分析关键区域，并以计算突变和跨物种迁移检验解释范围。\n来源：reports/final/PR02-01_REPORT.md';
}
// 02
{
 const s=slide('研究问题与实验设计','主任务为强度预测与位点解释，选做跨物种迁移和多方法归因',
 ['project-pool(2).pdf，PR02-01条目','reports/final/PR02-01_REPORT.md'],
 'RQ1使用共同样本对照R²和Spearman，并检查强度分组误差。RQ2比较归因位点与UP、−35、spacer和−10区域。RQ3使用等数量替换和单点方向检验。变体结果均为计算预测。');
 [['RQ1  预测性能','数据驱动模型与热力学模型各自表现如何？'],['RQ2  关键区域','模型重要位点是否集中于核心启动子区域？'],['RQ3  突变影响','改变高贡献位点后，预测强度是否按预期变化？']].forEach((r,i)=>{
  text(s,r[0],68,202+145*i,300,57,32,true,C.teal);text(s,r[1],374,205+145*i,830,88,30);
 });
}
// 03
{
 const s=slide('数据构建与评价划分','源域恢复TSS上下文，目标域保留教师提供的原始序列',
 ['data/03_Ecoli_strength87/source_manifest.json','data/04_Bsub_strength61/data_manifest.json','https://doi.org/10.1093/nar/gkaa325','https://doi.org/10.1128/JB.02096-14'],
 '大肠杆菌原始11,884条记录中，11,809条参考位置唯一，75条多位置命中记录排除。使用历史参考U00096.2双链定位并与Thomason原TSS表核对，保留原标签和继承划分。目标域591行对应580种序列，相同序列绑定分组。历史test曾用于项目早期分析，所有结果按内部基准解释。');
 table(s,['数据','有效记录与长度','训练 / 验证 / 测试'],[
  ['大肠杆菌','11,809条，87 bp','8,267 / 1,768 / 1,774'],['枯草杆菌','591行，61 bp','354 / 119 / 118']],{y:190,h:205,widths:[230,375,543],size:27});
 text(s,'原50 bp在U00096.2上定位，再恢复TSS上游67 bp及下游20 bp。',69,438,1128,70,29,true);
 text(s,'源域目标采用log10表达关联值。目标域单位、TSS和方向未知。',69,535,1128,78,28,false,C.muted);
}
// 04
{
 const s=slide('87 bp上下文与热力学预测','TSS位于零基索引67，固定窗口与逐序列推定窗口分别统计',
 ['contracts/CONTEXT87_V3.md','pr02/context87/experiment.py','https://doi.org/10.1038/s41467-022-32829-5'],
 '固定窗口依次为UP[7,31)、−35[32,38)、spacer[38,55)、−10[55,61)、discriminator[61,67)和初始转录区[67,87)。索引31为固定参考布局中的间隔。Promoter Calculator在核验TSS处枚举spacer15至20 bp和discriminator6至10 bp，共30种状态，选择最小自由能状态。原始Tx_rate使用train拟合的log10线性关系校准到表达尺度。固定窗口和工具推定盒区都没有逐条结合位点实测注释。');
 table(s,['固定区域','相对TSS位置','长度'],[
 ['UP','−60至−37','24 bp'],['−35盒','−35至−30','6 bp'],['spacer','−29至−13','17 bp'],['−10盒','−12至−7','6 bp'],['discriminator','−6至−1','6 bp'],['初始转录区','+1至+20','20 bp']],{x:66,y:186,w:670,h:414,widths:[245,260,165],size:25});
 text(s,'30种状态',797,206,406,66,43,true,C.teal);
 text(s,'spacer 15至20 bp\ndiscriminator 6至10 bp',799,301,403,109,28);
 text(s,'最小自由能状态\n训练集拟合尺度校准',799,462,403,106,28);
 text(s,'表中为分析窗口，另含7 bp上游侧翼和−36处1 bp间隔。盒区采用固定假设或模型推定。',69,630,1134,36,22,false,C.muted);
}
// 05
{
 const s=slide('训练集分布与区域特征','描述统计仅使用8,267条训练序列',
 ['data/03_Ecoli_strength87/samples.tsv','runs/ecoli87_m3_completion_20261013_v1/source_tables/spacing_correlations.csv'],
 '原始表达具有较长右尾，使用log10尺度拟合。固定−10窗口平均GC为0.2326，但其共识匹配与表达的Spearman仅−0.0221。固定−35匹配为0.0189。工具推定spacer17 bp占52.38%，长度与表达Spearman为−0.0008。单一局部描述符不足以可靠排序，不能据此排除组合效应。');
 const bins=Array(12).fill(0);for(const r of train){const i=Math.max(0,Math.min(11,Math.floor((Number(r.target_log10)-.5)/.5)));bins[i]++;}
 chart(s,Array.from({length:12},(_,i)=>f(.5+(i+.5)*.5,2)),[{name:'训练序列数',values:bins}],{x:66,y:203,w:554,h:341,ylabel:'序列数',format:'0',ymin:0,legend:false,labelsize:18,gap:6});
 text(s,'log10表达，分箱宽度0.5',135,566,480,40,22,false,C.muted);
 const eda=report.checks.eda.filter(r=>['全序列 GC','−10 共识匹配','−35 共识匹配'].includes(r.feature));
 table(s,['特征','Spearman'],eda.map(r=>[r.feature,f(r.spearman)]),{x:684,y:213,w:520,h:257,widths:[330,190],size:25});
 text(s,'单个共识匹配度\n与表达的单调关系较弱',685,514,520,98,28,true,C.teal);
}
// 06
{
 const s=slide('模型与共同评价尺度','全部模型使用相同87 bp输入范围，预处理仅拟合train',
 ['configs/m4_protocol_v1.json','configs/m4_refinement_v2.json','https://arxiv.org/abs/2306.15006'],
 'CNN四组配置各训练三个种子，以平均验证MSE选择C0。C0为两层卷积32/64通道、核7/5、池化长度5、隐藏层32、dropout0.3、weight decay0.001，最多100轮，patience15。DNA采用官方DNABERT-2-117M固定revision，冻结编码器，平均有效序列token得到768维表征。所有R²、RMSE、MAE在log10尺度计算，Spearman评价排序。验证集参与选参。');
 table(s,['模型','表示或机制','拟合方式'],[
  ['KNN','理化特征','距离加权回归'],['Ridge / LinearSVR','归一化k-mer频率','train标准化，val选参'],['CNN','独热编码，两层卷积','四配置，三个种子'],['DNA + Ridge','冻结DNABERT-2，768维','编码器固定，训练回归头'],['Promoter Calculator','转录起始热力学','固定TSS，train校准']],{y:190,h:352,widths:[285,460,403],size:25});
 text(s,'共同验证集1,768条，覆盖率100%。CNN先逐种子计分，再报告均值与SD。',69,579,1129,68,27,false,C.muted);
}
// 07
{
 const s=slide('CNN与k-mer回归优于校准热力学模型','共同验证集的内部比较，Ridge与DNA头采用扩展正则化范围',
 [R0+'/evaluation/model_summary.csv',R1+'/regularization/summary.csv'],
 'KNN、LinearSVR、Promoter Calculator和CNN保留原训练设计。主表更新k-mer和DNA的Ridge头。CNN的R²为0.115967±0.007110，Spearman0.307694±0.014280。训练均值R²接近0，Spearman未定义。该验证集用于选参，分数属于内部描述性比较，不能当作外部盲测成绩。');
 const methods=[old.knn,refined.kmer_ridge,old.linear_svr,refined.dnabert2_ridge,old.thermo,old.cnn];
 chart(s,['KNN','Ridge','SVR','DNA','热力学','CNN'],[{name:'验证R²',values:methods.map(r2),points:methods.map((_,i)=>({idx:i,fill:i===5?C.teal:i===4?C.orange:C.blue}))}],{ymin:0,ymax:.145,ylabel:'R²，log10尺度',legend:false,labels:true,w:775,h:390});
 finding(s,'0.1160','CNN三种子平均R²\nSD = 0.0071',{x:896,y:213});
 text(s,'Spearman 0.3077',897,442,310,58,29,true,C.teal);
 text(s,'预测仍只能解释\n有限的表达方差',897,534,310,90,27,false,C.muted);
}
// 08
{
 const s=slide('正则化改善融合表现，新增表征未持续增益','固定输入表征与CNN种子，Ridge的alpha由不超过100扩展至1,000,000',
 [R0+'/ablation/stage_summary.csv',R1+'/regularization/summary.csv',R1+'/regularization/search.csv'],
 'A0为256维k-mer，A1增加320维CNN池化表征，A2增加768维冻结DNA，A3增加训练IG选择区域的特征。A1至A3均以三个配对种子计分。所有11个源域回归头选择内部候选alpha10000。A3的平均R²从−0.027259变为0.114434。扩展后A2低于A1，A3有所回升但未超过A1，未提供稳定递增收益的证据。');
 const stages=['A0','A1','A2','A3'];
 chart(s,stages,[{name:'原alpha范围',values:stages.map(k=>r2(oldStages[k])),fill:C.gray},{name:'扩展alpha范围',values:stages.map(k=>r2(refined[k==='A0'?'kmer_ridge':k])),fill:C.teal}],{ymin:-.05,ymax:.15,major:.05,ylabel:'验证R²',w:780,h:391});
 text(s,'A0  k-mer\nA1  加入CNN表征\nA2  加入冻结DNA\nA3  加入归因区域',897,208,310,205,27);
 text(s,'A1：0.1174',897,449,310,58,34,true,C.teal);
 text(s,'A2、A3均未超过A1',897,542,310,84,27,false,C.muted);
}
// 09
{
 const s=slide('基因组区域分组后，整体解释度仍有限','20 kb区域分组，跨集合双链窗口重叠为零',
 [R1+'/grouped/split_audit.json',R1+'/grouped/summary.csv'],
 '新train/dev/evaluation为5985/2003/1998条，仅使用原train与val，排除49条跨区域边界窗口及1774条历史test。相同序列及其反向互补副本绑定分组。所有监督预处理、CNN、热力学校准和归因选区重新拟合新train，以dev早停与选参。该设计是同一基因组一次内部空间划分，历史标签曾参与项目开发。与原验证集的组成不同，不能用分数差直接估计泄漏影响。');
 const order=['kmer_ridge','dnabert2_ridge','cnn','A1','A2','A3'];
 chart(s,['Ridge','DNA','CNN','A1','A2','A3'],[{name:'区域分组R²',values:order.map(k=>r2(spatial[k])),points:order.map((k,i)=>({idx:i,fill:k==='A1'?C.teal:C.blue}))}],{ymin:0,ymax:.145,ylabel:'evaluation R²',w:775,legend:false,labels:true,h:390});
 text(s,'5,985 / 2,003 / 1,998',896,210,320,100,31,true,C.teal);
 text(s,'train / dev / evaluation',896,316,320,77,24,false,C.muted);
 text(s,'A1：0.1162',896,430,320,68,35,true);
 text(s,'一次内部空间对照\n尚非外部独立验证',896,531,320,95,27,false,C.muted);
}
// 10
{
 const s=slide('−10区域在三个CNN种子中保持富集','IG为积分梯度，ISM为逐位点替换。固定窗口Top-5，64条母序列',
 [R1+'/attribution_stability/region_by_seed.csv',R1+'/attribution_stability/region_across_seeds.csv','https://proceedings.mlr.press/v70/sundararajan17a.html'],
 '同一64条验证母序列，IG采用均匀0.25基线与64步梯形积分。ISM为每个位点三种替换的平均绝对log10变化。在IG与ISM、固定及PC推定窗口、Top5及Top10全部组合下，−10三个种子的平均富集均为正。均匀随机位置是描述性参照，未消除TSS对齐和模型位置偏好，也未进行多重比较校正。');
 const names=['UP','-35','spacer','-10'];
 for(const [j,method] of ['integrated_gradients','ism_mean_absolute'].entries()){
  text(s,j?'ISM':'IG',j?697:153,180,400,42,29,true,C.teal);
  const series=protocol.seeds.map((seed,i)=>({name:String(seed),values:names.map(name=>Number(region.find(r=>r.seed===String(seed)&&r.method===method&&r.region===name&&r.window_policy==='fixed'&&r.top_k==='5').mean_excess_fraction)),fill:[C.blue,C.orange,C.teal][i]}));
  chart(s,names,series,{x:j?660:66,y:229,w:550,h:332,ymin:-.2,ymax:.25,format:'0.00',labelsize:23,gap:55});
 }
 text(s,'命中比例扣除区域长度/87。两类窗口与Top-5/10均支持−10富集。',69,600,1135,62,27,true);
}
// 11
{
 const s=slide('精确位点的一致性弱于区域层面','同一64条序列，比较归因方法差异与CNN训练随机性',
 [R0+'/explanation_analysis/consistency_summary.csv',R1+'/attribution_stability/cross_seed_summary.csv'],
 '同一CNN的IG与ISM位点排序相关为0.5105，Top5 Jaccard0.3677。跨种子IG相关0.2861，ISM相关0.4242。跨种子Top5 Jaccard分别0.2024和0.2522。区域富集一致与单个位点清单不稳定可以同时存在。IG沿连续基线积分，ISM评价离散替换，定义差异也会影响局部排序。');
 const within5=Number(within.find(r=>r.metric==='top_k_jaccard'&&r.top_k==='5').mean);
 const cross=method=>Number(across.find(r=>r.method===method&&r.metric==='top_k_jaccard'&&r.top_k==='5').mean);
 chart(s,['同一模型\nIG与ISM','跨种子\nIG','跨种子\nISM'],[{name:'Top-5 Jaccard',values:[within5,cross('integrated_gradients'),cross('ism_mean_absolute')],points:[{idx:0,fill:C.blue},{idx:1,fill:C.orange},{idx:2,fill:C.teal}]}],{ymin:0,ymax:.5,legend:false,labels:true,ylabel:'Top-5集合Jaccard',w:770,h:385});
 text(s,'跨种子仅约\n0.20至0.25',896,215,310,130,36,true,C.teal);
 text(s,'单次Top-k清单\n难以确定碱基的\n真实生物作用',897,407,310,160,28);
}
// 12
{
 const s=slide('高归因位点替换引起更大的预测变化','64条母序列，每组5个位点、10次替换，共1,920个变体',
 [R0+'/mutation/mutation_summary.csv',R0+'/mutation/paired_amplitude.csv'],
 'High和Low分别选择IG最高和最低5位。Random从两组之外抽取5位，所有替换均改变原碱基。先在母序列内汇总，再按母序列bootstrap。High−Random平均绝对变化差0.05995，95%区间0.03673至0.08319。High有符号均值接近0，重要位点的随机替换不预设增强或减弱。这些结果衡量代表CNN的输出敏感性。');
 chart(s,['High','Low','Random'],[{name:'平均绝对变化',values:mutation.map(r=>Number(r.mean_absolute_delta_log10)),points:[{idx:0,fill:C.teal},{idx:1,fill:C.gray},{idx:2,fill:C.blue}]}],{x:66,y:202,w:600,h:350,ymin:0,ymax:.17,ylabel:'|预测变化|，log10',legend:false,labels:true});
 table(s,['组','95%母序列区间'],mutation.map(r=>[r.group,f(r.ci_low)+'至'+f(r.ci_high)]),{x:736,y:207,w:470,h:267,widths:[115,355],size:24});
 text(s,'High与Random的配对差\n0.0600 [0.0367, 0.0832]',738,507,466,93,27,true,C.teal);
 text(s,'高归因位置对模型更敏感，真实表达效应仍需测量。',69,620,1125,40,27);
}
// 13
{
 const s=slide('梯度与CNN的单点变化方向高度一致','High 5位穷举3种替换，共960个单点变体',
 [R0+'/mutation/direction_summary.csv'],
 '预期变化来自野生型梯度的替换通道减原通道。近零阈值10的负4次方log10。预期正向477条、负向483条。梯度与CNN一致率98.229%，95%母序列聚类区间97.292%至99.063%。CNN与热力学一致率44.271%。这反映模型局部自洽及预测函数分歧，缺少实测变体标签，不能证明生物因果指导作用。');
 chart(s,['梯度与CNN','梯度与热力学','CNN与热力学'],[{name:'方向一致率',values:['cnn_agreement','thermo_agreement','cross_model_agreement'].map(k=>Number(direction[k])),points:[{idx:0,fill:C.teal},{idx:1,fill:C.gray},{idx:2,fill:C.orange}]}],{ymin:0,ymax:1.12,format:'0.0%',legend:false,labels:true,w:775,h:390});
 finding(s,pct(direction.cnn_agreement),'梯度与CNN有限替换\n方向一致率',{x:896,y:213});
 text(s,pct(direction.cross_model_agreement),897,446,310,66,40,true,C.orange);
 text(s,'CNN与热力学方向\n存在明显分歧',897,532,310,90,27,false,C.muted);
}
// 14
{
 const s=slide('跨物种few-shot未显示稳定收益','目标test 118行，五次固定训练组抽样，目标回归头仅使用train和val',
 [R1+'/transfer_stability/summary.csv',R1+'/transfer_stability/metrics.csv'],
 '每次排列内20/50/100训练组嵌套，共30项few-shot拟合及2项zero-shot。标准差来自五次训练组抽样，未代表所有泛化不确定性。100组冻结CNN的Spearman均值0.20747±0.01617，zero-shot CNN0.21609。旧单次抽样约0.263还使用不同正则化范围，两组变化不能只归于抽样。目标单位未知，zero-shot绝对误差只作诊断。目标test曾用于项目历史分析。');
 const get=c=>transfer.find(r=>r.condition===c);
 const series=[['kmer','k-mer',C.blue],['frozen_cnn','CNN',C.teal]].map(([key,name,fill])=>({name:name+' few-shot',fill,values:[20,50,100].map(n=>rho(get(`few_shot_${key}_${n}_groups`)))}));
 series.push({name:'CNN zero-shot',fill:C.orange,values:[20,50,100].map(()=>rho(get('zero_shot_cnn'))),line:{fill:C.orange,width:2,style:'dashed'}});
 chart(s,['20','50','100'],series,{type:'line',x:66,y:205,w:700,h:354,ymin:0,ymax:.3,ylabel:'目标test Spearman'});
 text(s,'目标训练序列组数',222,584,520,40,23,false,C.muted);
 table(s,['组数','CNN均值 ± SD'],[20,50,100].map(n=>{const r=get(`few_shot_frozen_cnn_${n}_groups`);return[String(n),f(r.spearman_mean)+' ± '+f(r.spearman_sd)];}),{x:814,y:217,w:392,h:250,widths:[94,298],size:24});
 text(s,'100组：0.2075\nzero-shot：0.2161',815,512,392,102,28,true,C.teal);
}
// 15
{
 const s=slide('结果的解释范围','预测关联、模型敏感性与生物因果分别解释',
 ['reports/final/PR02-01_REPORT.md，第7、12节'],
 '源域标签来自天然转录组，还包含细胞状态及转录后过程的影响。热力学模型针对转录起始率，其拟合较弱不能单独归因于数据或模型。中强度组热力学MAE0.1723低于CNN0.2003，但仍高于训练均值基线0.1644。原验证集参与选参，空间分组只有一次，历史源域及目标test曾被观察，结论限定内部评价。归因富集采用描述性区间，未作多重比较校正。');
 const rows=[['标签与机制目标','天然表达关联值包含序列之外因素，热力学预测针对转录起始。'],['内部评价','验证集参与选参，空间分组仅一次，历史test已被观察。'],['归因证据','区域富集较一致，精确位点和真实突变效果仍有不确定性。']];
 rows.forEach((r,i)=>{text(s,r[0],68,206+i*132,290,54,31,true,C.teal);text(s,r[1],383,207+i*132,820,90,28);});
 text(s,'中强度组中，常数基线比热力学和CNN的MAE更低。',69,620,1127,42,25,false,C.muted);
}
// 16
{
 const s=slide('研究结论','四项必做实验及跨物种、多方法归因两项选做实验',
 ['reports/final/PR02-01_REPORT.md','https://github.com/Azusakada/PR02-01-promoter--insight'],
 'RQ1：CNN和k-mer表征在本数据上优于校准热力学，整体R²约0.1。正则化显著影响融合结果，增加DNA和归因特征没有稳定递增收益。RQ2：−10区域富集跨方法和种子较一致，精确碱基位置不稳定。RQ3：高归因位置更敏感，梯度较好预测CNN单点方向，但缺少变体实测标签，生物因果仍需独立验证。数据、预测、源码、报告及重建入口保存在项目仓库。');
 [['RQ1','数据驱动模型有一定预测能力，整体解释度仍有限。'],['RQ2','−10区域信号较一致，精确位点需要更多验证。'],['RQ3','高归因位置更敏感，真实突变效果仍需实测。']].forEach((r,i)=>{
  text(s,r[0],69,209+i*119,135,58,34,true,C.teal);text(s,r[1],247,211+i*119,955,85,30);
 });
 text(s,'代码、数据与完整报告',69,599,1100,38,24,true);
 text(s,'github.com/Azusakada/PR02-01-promoter--insight',69,643,1100,35,22,false,C.muted);
}
// 17
{
 const s=slide('备查：不同强度组的误差','分组阈值来自源域train三分位，MAE均在log10尺度',
 ['reports/final/report_content.json，checks.group_mae'],
 'CNN先按三个种子分别计算组内MAE，再取均值。k-mer Ridge使用扩展正则化头。中强度样本的标签接近训练均值，因此常数预测的组内MAE较低，不能把热力学低于CNN解释为机制优势。');
 table(s,['组','样本数','CNN','Ridge','热力学','均值基线'],report.checks.group_mae.map(r=>[r.group,String(r.n),f(r.cnn_mae),f(r.ridge_mae),f(r.pc_mae),f(r.mean_mae)]),{y:212,h:290,widths:[180,135,205,205,205,218],size:26});
 text(s,'低、高强度组中CNN的MAE更低。中强度组的常数参照最好。',69,556,1124,86,29,true,C.teal);
}
// 18
{
 const s=slide('备查：两种划分的完整性能','不同样本组成的内部评价，分数差不直接等于重叠窗口造成的偏差',
 [R0+'/evaluation/model_summary.csv',R1+'/regularization/summary.csv',R1+'/grouped/summary.csv'],
 '表格为log10尺度R²及Spearman的均值。CNN和A1至A3分别保留三个种子，标准差见完整报告。训练均值Spearman未定义。原划分val为1768条，空间evaluation为1998条。两个集合不能视作同一批样本的严格配对泛化比较。');
 const keys=['knn','kmer_ridge','linear_svr','dnabert2_ridge','thermo','cnn','A1','A2','A3','train_mean'];
 const names=['KNN','k-mer Ridge','LinearSVR','DNA + Ridge','热力学','CNN','A1','A2','A3','训练均值'];
 const rows=keys.map((k,i)=>{const a=refined[k]||old[k],b=spatial[k];return[names[i],f(a.r2_log10_mean),a.spearman_mean?f(a.spearman_mean):'未定义',f(b.r2_log10_mean),b.spearman_mean?f(b.spearman_mean):'未定义'];});
 table(s,['方法','原划分R²','原划分ρ','空间R²','空间ρ'],rows,{y:180,h:454,widths:[276,218,218,218,218],size:24,highlight:[5,6],padding:5});
}
// 19
{
 const s=slide('备查：数据与方法来源','完整书目信息与链接见研究报告参考文献',
 report.references.map(r=>r.text+'\n'+r.url),
 '原始数据与TSS定位分别引用Wang2020和Thomason2015，冻结DNA表征引用DNABERT-2，热力学模型引用LaFleur2022，Integrated Gradients引用Sundararajan2017。枯草杆菌CSV由课程教师提供，论文、测量单位、TSS和方向信息未附。');
 const refs=[['Wang等，2020，Nucleic Acids Research','Deep_promoter原始数据与代码'],['Thomason等，2015，Journal of Bacteriology','大肠杆菌TSS图谱'],['LaFleur等，2022，Nature Communications','Promoter Calculator'],['Zhou等，2024，ICLR','DNABERT-2冻结表征'],['Sundararajan等，2017，PMLR','Integrated Gradients']];
 refs.forEach((r,i)=>{text(s,r[0],69,194+i*79,1150,38,25,true);text(s,r[1],69,233+i*79,1150,36,23,false,C.muted);});
 text(s,'枯草杆菌数据：课程教师提供的61 bp强度CSV。',69,624,1128,40,24);
}

if(p.slides.items.length!==19)throw new Error('Unexpected slide count');
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
// Use project metadata and omit editing-session identity from the document package.
execFileSync(python,['-c',String.raw`import sys,zipfile
from xml.etree import ElementTree as E
path=sys.argv[1]
with zipfile.ZipFile(path) as z: parts={n:z.read(n) for n in z.namelist()}
core=E.fromstring(parts['docProps/core.xml'])
fields={'{http://purl.org/dc/elements/1.1/}creator':'PR02-01 项目组','{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy':'','{http://purl.org/dc/elements/1.1/}title':'启动子强度预测与关键位点归因','{http://purl.org/dc/elements/1.1/}description':''}
for tag,value in fields.items():
 node=core.find(tag)
 if node is None: node=E.SubElement(core,tag)
 node.text=value
parts['docProps/core.xml']=E.tostring(core,encoding='utf-8',xml_declaration=True)
if 'docProps/app.xml' in parts:
 app=E.fromstring(parts['docProps/app.xml'])
 for node in app:
  if node.tag.rsplit('}',1)[-1] in ['Application','AppVersion','Company','Manager','Template']:node.text=''
 parts['docProps/app.xml']=E.tostring(app,encoding='utf-8',xml_declaration=True)
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
 for name,value in parts.items():z.writestr(name,value)
`,candidate]);
await fs.writeFile(path.join(build,'source_evidence.json'),JSON.stringify({report_revision:'0b8ca2cc745b3289aad2738aaeb14d37a092f287',sources,titles,evidence},null,2)+'\n');
const result=await finalizePresentation({workspaceDir:repo,candidatePath:candidate,finalPath,pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--cover-role','cover','--validate-bullet-geometry','--validate-heading-fit',...new Set(nativeTables)].flatMap(v=>typeof v==='number'?['--require-native-table-slide',String(v)]:[v]),
 explicitTotalSlideCount:19,requiredNativeTableOwnerSlides:[...new Set(nativeTables)],requiredNativeChartOwnerSlides:[...new Set(nativeCharts)],
 fontPolicy:{basis:'design',families:[FONT]},materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation-'+path.basename(finalPath)+'.json')});
// Keep the chart packager's temporary files in the private build directory.
for(const entry of await fs.readdir(repo,{withFileTypes:true})){
 if(!entry.isDirectory()||!entry.name.startsWith('.chart-data-'))continue;
 const temporary=path.join(repo,entry.name);
 let snapshot;
 try{snapshot=JSON.parse(await fs.readFile(path.join(temporary,'chart-data-snapshot.json'),'utf8'));}
 catch(error){if(error.code==='ENOENT'||error instanceof SyntaxError)continue;throw error;}
 if(snapshot.output_sha256===result.finalSha256)await fs.rename(temporary,path.join(build,entry.name.slice(1)));
}
const imported=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<imported.slides.items.length;i++){
 const png=await imported.export({slide:imported.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(JSON.stringify({finalPath,slideCount:19,chartCount:nativeCharts.length,tableCount:nativeTables.length,receiptPath:result.receiptPath},null,2));
