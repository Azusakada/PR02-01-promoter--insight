// Editable M5 charts and tables, authored with the Codex bundled Artifact Tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';

const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const modules=process.env.PR02_NODE_MODULES;
const skill=process.env.PR02_PRESENTATIONS_SKILL;
if(!modules||!skill)throw new Error('Set PR02_NODE_MODULES and PR02_PRESENTATIONS_SKILL to the bundled runtime and skill directories.');
process.env.RUNTIME_NODE_MODULES ||= modules;
const resolve=createRequire(path.join(modules,'entry.cjs'));
const {Presentation,PresentationFile}=await import(pathToFileURL(resolve.resolve('@oai/artifact-tool')).href);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const content=JSON.parse(await fs.readFile(path.join(repo,'reports/M5_REPORT_CONTENT.json'),'utf8'));
const root=path.join(repo,content.run_id);
const temp=path.join(repo,'work/m5_slides_release');await fs.mkdir(temp,{recursive:true});
function parseCsv(text){
 const rows=[];let row=[],field='',quoted=false;
 for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}
 else if(c===','&&!quoted){row.push(field);field='';}else if(c==='\n'&&!quoted){row.push(field);rows.push(row);row=[];field='';}else if(c!=='\r')field+=c;}
 if(field||row.length){row.push(field);rows.push(row);}const header=rows.shift();return rows.filter(r=>r.length===header.length).map(r=>Object.fromEntries(header.map((h,i)=>[h,r[i]])));
}
async function csv(p){return parseCsv(await fs.readFile(path.join(root,p),'utf8'));}
const models=await csv('evaluation/model_summary.csv');const mm=Object.fromEntries(models.map(r=>[r.method,r]));
const grid=await csv('cnn/search.csv');const stages=await csv('ablation/stage_summary.csv');
const overlap=await csv('explanation_analysis/overlap_summary.csv');const agreement=await csv('explanation_analysis/consistency_summary.csv');
const mutation=await csv('mutation/mutation_summary.csv');const direction=(await csv('mutation/direction_summary.csv')).find(r=>r.expected_direction==='all');
const transfer=await csv('transfer/transfer_metrics.csv');const groups=await csv('evaluation/group_metrics.csv');
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='Source Han Sans CN',ink='#183047',muted='#51616F',blue='#386A91',orange='#D88B2E',green='#679C87';
const f=(v,n=3)=>Number(v).toFixed(n), pct=v=>(100*Number(v)).toFixed(1)+'%';
function text(s,value,x,y,w,h,size=28,bold=false,color=ink){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=value;t.text.style={typeface:font,fontSize:size,bold,color,verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0,lineSpacing:1.22};return t;
}
function slide(title,source,note=''){
 const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,70,45,1140,72,44,true);
 if(source)text(s,'来源：'+source,72,665,1090,35,15,false,muted);
 text(s,String(p.slides.items.length).padStart(2,'0'),1170,665,45,25,16,false,muted);
 s.speakerNotes.text=note+'\n运行身份：'+content.run_id+'\n源表：'+source;
 return s;
}
function chart(s,categories,series,{ytitle='',ymin,ymax,format='0.000',left=72,top=166,width=760,height=435,type='bar',legend=true}={}){
 const c=s.charts.add(type,{position:{left,top,width,height},categories,series:series.map((r,i)=>({name:r.name,values:r.values.map(v=>Number(Number(v).toFixed(8))),fill:[blue,orange,green][i%3]})),
  hasLegend:legend,legend:{position:'bottom',textStyle:{fontSize:19}},barOptions:{direction:'column',grouping:'clustered',gapWidth:100},
  xAxis:{textStyle:{fontSize:19},line:{fill:'#A6AFB7',width:1}},
  yAxis:{title:{text:ytitle,textStyle:{fontSize:19}},numberFormatCode:format,min:ymin,max:ymax,textStyle:{fontSize:18},majorGridlines:{fill:'#DCE2E7',width:1}},
  chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}});
 applyPresentationChartFont(c,{fontFamily:font});return c;
}
function table(s,headers,rows,{left=72,top=160,width=1136,height=360,columnWidths}={}){
 const t=s.tables.add({rows:rows.length+1,columns:headers.length,left,top,width,height,values:[headers,...rows],columnWidths});
 t.borders.assign({fill:'#D9D9D9',width:1,style:'solid'});
 for(let i=0;i<rows.length+1;i++)for(let j=0;j<headers.length;j++){
  const c=t.getCell(i,j);c.fill=i===0?'#DDE7F0':i%2===0?'#F4F7F9':'#FFFFFF';
  c.text.style={typeface:font,fontSize:23,bold:i===0,color:ink,alignment:j===0?'left':'center',verticalAlignment:'middle',insets:{left:12,right:12,top:8,bottom:8}};
 }
 return t;
}
// 1
{
 const s=p.slides.add();s.background.fill='#F5F8FA';
 text(s,'启动子强度预测\n与位点解释',90,155,1100,200,66,true);
 text(s,'PR02-01 项目最终答辩',94,392,1000,52,33,false,blue);
 text(s,'胡昊铭  李宇飞  李玘航  田惠今  陈思远',94,498,1080,48,25);
 text(s,'2026年10月14日',94,571,900,40,24,false,muted);
 s.speakerNotes.text='本答辩基于'+content.run_id+'的冻结结果。数据、方法和限制均与最终报告一致。';
}
// 2
{
 const s=slide('研究问题与证据范围','课程 project-pool(2).pdf 第45至47页');
 text(s,'RQ1  哪类模型更适合当前表达标签？',75,170,1100,60,33,true);
 text(s,'统一87bp输入与验证ID，对照数据驱动模型和热力学模型。',75,238,1100,55,27);
 text(s,'RQ2  重要位点是否对应启动子区域？',75,330,1100,60,33,true);
 text(s,'比较IG、ISM的Top-k与固定及推定区域，校正区域长度。',75,398,1100,55,27);
 text(s,'RQ3  归因能否预测突变影响？',75,490,1100,60,33,true);
 text(s,'检验计算敏感性与变化方向，区分模型自洽和生物因果。',75,558,1100,55,27);
}
// 3
{
 const s=slide('数据与统一评价协议','protocol/source_audit.json；CONTEXT87_V3.md');
 table(s,['数据','记录与长度','训练 / 验证 / 测试'],[['E. coli','11809条，87bp','8267 / 1768 / 1774'],['B. subtilis','591行，61bp','354 / 119 / 118']],{height:215,columnWidths:[200,410,526]});
 text(s,'源域TSS索引67，原50bp位于[17,67)。',75,417,1120,48,28);
 text(s,'监督拟合只用train，val按预定规则选参。源域test本轮不评价。',75,485,1120,62,27);
 text(s,'目标域重复序列按组划分。单位、TSS和方向仍未知。',75,563,1120,55,27,false,muted);
}
// 4
{
 const s=slide('六类模型的验证表现','evaluation/model_summary.csv','CNN误差条见报告，三种子R²样本标准差0.0071。所有模型同一1768条val，覆盖率100%。');
 const order=['knn','ridge','linear_svr','dnabert2_ridge','thermo','cnn'];
 chart(s,['KNN','Ridge','LinSVR','DNA','PC','CNN'],[{name:'R²',values:order.map(k=>Number(mm[k].r2_log10_mean))}],{ytitle:'R²（log10）',ymin:0,ymax:.14,legend:false});
 text(s,'CNN',886,182,310,52,35,true);text(s,f(mm.cnn.r2_log10_mean,3)+' ± '+f(mm.cnn.r2_log10_sd,3),886,250,315,65,40,true,blue);
 text(s,'三个种子的R²均值 ± SD',886,329,315,70,23,false,muted);
 text(s,'Spearman '+f(mm.cnn.spearman_mean,3),886,425,315,56,28,true);
 text(s,'准确率仍有限。',886,514,315,54,28);
}
// 5
{
 const s=slide('CNN配置的三种子对照','cnn/search.csv','四配置各3个种子，按平均val MSE选C0。固定种子20260928用于归因和迁移。');
 const cats=['C0','C1','C2','C3'];const values=cats.map(c=>Math.sqrt(grid.filter(r=>r.condition===c).reduce((a,r)=>a+Number(r.val_mse_log10),0)/3));
 chart(s,cats,[{name:'三种子平均MSE的平方根',values}],{ytitle:'验证RMSE',ymin:.578,ymax:.590,legend:false});
 text(s,'C0保留原架构',885,186,315,70,31,true);
 text(s,'pool 5\ndropout 0.3\nweight decay 0.001',885,285,320,175,27);
 text(s,'更大池化长度和更低正则化未稳定改善。',885,498,320,112,26,false,muted);
}
// 6
{
 const s=slide('热力学模型的适用范围','evaluation/group_metrics.csv；M3 calibration.json','低/中/高组阈值来自train三分位。CNN为3个种子平均MAE。');
 function groupMean(method,g){const r=groups.filter(r=>r.method===method&&r.subset==='val'&&r.intensity_group===g);return r.reduce((a,r)=>a+Number(r.mae_log10),0)/r.length;}
 chart(s,['低强度','中强度','高强度'],[{name:'CNN',values:['low','middle','high'].map(g=>groupMean('cnn',g))},{name:'热力学',values:['low','middle','high'].map(g=>groupMean('thermo',g))}],{ytitle:'组内MAE（log10）',ymin:0,ymax:.75});
 text(s,'热力学可用',886,181,315,62,32,true);
 text(s,'已核验TSS支持\n30种状态推断\n及训练集校准。',886,277,320,143,27);
 text(s,'天然RNA表达与转录起始率的测量目标不同。',886,459,320,137,27,false,muted);
}
// 7
{
 const s=slide('−10窗口具有更一致的Top-k富集','explanation_analysis/overlap_summary.csv','图为固定窗口Top-5。两种窗口政策与95%母序列bootstrap区间见报告。未做多重比较校正。');
 const regions=['UP','-35','spacer','-10'];
 chart(s,regions,['integrated_gradients','ism_mean_absolute'].map((method,i)=>({name:i?'ISM':'IG',values:regions.map(name=>Number(overlap.find(r=>r.method===method&&r.region===name&&r.top_k==='5'&&r.annotation_status==='canonical_window_assumption').mean_excess_fraction))})),{ytitle:'命中比例减区域长度/87',ymin:-.18,ymax:.18});
 text(s,'比较扣除区域长度',886,180,318,85,29,true);
 text(s,'IG与ISM都显示−10正向富集。',886,296,320,110,29);
 text(s,'其余区域没有一致的正向证据。盒区仍是模型假设或推定。',886,458,320,150,26,false,muted);
}
// 8
{
 const s=slide('IG与ISM部分一致','explanation_analysis/consistency_summary.csv','同一CNN、同一64条val；Spearman对87位点绝对重要性排名计算。区间按母序列bootstrap。');
 const vals=[agreement.find(r=>r.top_k==='5'&&r.metric==='spearman'),agreement.find(r=>r.top_k==='5'&&r.metric==='top_k_jaccard'),agreement.find(r=>r.top_k==='10'&&r.metric==='top_k_jaccard')];
 table(s,['指标','均值','95%区间'],vals.map((r,i)=>[["位点排序Spearman","Top-5 Jaccard","Top-10 Jaccard"][i],f(r.mean),f(r.ci_low)+' 至 '+f(r.ci_high)]),{height:300,columnWidths:[430,220,486]});
 text(s,'归因方法会影响具体Top-k位点选择。',75,505,1120,55,32,true);
 text(s,'最低及最高Jaccard各3条案例按固定规则保留，避免只展示符合预期的样本。',75,578,1120,58,25,false,muted);
}
// 9
{
 const s=slide('逐步加入特征没有改善R²','ablation/stage_summary.csv','A0 kmer；A1 +CNN；A2 +DNA；A3 +训练IG选区。统一StandardScaler(train)+Ridge头，val选alpha。A1/A2/A3按3种子配对。');
 chart(s,['A0','A1','A2','A3'],[{name:'验证R²',values:stages.map(r=>Number(r.r2_log10_mean))}],{ytitle:'R²（log10）',ymin:-.05,ymax:.10,legend:false});
 text(s,'A0 '+f(stages[0].r2_log10_mean),886,183,315,57,33,true);
 text(s,'A3 '+f(stages[3].r2_log10_mean),886,267,315,57,33,true,orange);
 text(s,'新增表示可能引入冗余\n和估计方差。\n原因仍待检验。',886,377,320,140,27);
 text(s,'融合与纯CNN分别报告。',886,550,320,66,25,false,muted);
}
// 10
{
 const s=slide('高归因位点的突变影响更大','mutation/mutation_summary.csv；paired_amplitude.csv','High/Low/Random均为5位点、10次替换、64母序列，共1920变体。图为母序列汇总均值。High-Low与High-Random差的95%母序列bootstrap区间均为正。');
 chart(s,['High','Low','Random'],[{name:'平均绝对变化',values:mutation.map(r=>Number(r.mean_absolute_delta_log10))}],{ytitle:'平均 |delta|（log10）',ymin:0,ymax:.17,legend:false});
 text(s,'相同突变数量',886,183,315,65,32,true);
 text(s,'先按母序列汇总，再重采样64条母序列。',886,283,320,115,28);
 text(s,'随机改变重要位点也可能增强预测。High的有符号均值接近0。',886,453,320,150,26,false,muted);
}
// 11
{
 const s=slide('局部方向自洽不等于生物因果','mutation/direction_summary.csv','预期方向来自野生型梯度alt-channel minus ref-channel。High5 x 3替换 x 64=960单点变体。近零阈值1e-4 log10。');
 chart(s,['梯度与CNN','梯度与热力学','CNN与热力学'],[{name:'方向一致率',values:['cnn_agreement','thermo_agreement','cross_model_agreement'].map(k=>Number(direction[k]))}],{ytitle:'一致比例',ymin:0,ymax:1.05,format:'0%',legend:false});
 text(s,pct(direction.cnn_agreement),886,181,315,66,46,true,blue);text(s,'梯度与CNN有限替换',886,259,320,80,25);
 text(s,pct(direction.cross_model_agreement),886,382,315,66,46,true,orange);text(s,'CNN与热力学方向',886,460,320,65,25);
 text(s,'尚无变体的实测强度。',886,567,320,48,25,false,muted);
}
// 12
{
 const s=slide('跨物种迁移的效果有限','transfer/transfer_metrics.csv','目标test n=118行，置信区间按序列组bootstrap。目标test已有历史使用，本轮为同基准扩展。zero-shot绝对单位不可直接比较。');
 chart(s,['Zero-R','Zero-C','K20','C20','K50','C50','K100','C100'],[{name:'目标Spearman',values:transfer.map(r=>Number(r.spearman))}],{ytitle:'目标test Spearman',ymin:-.1,ymax:.4,legend:false});
 text(s,'最佳观察点约0.263',886,183,320,95,30,true);
 text(s,'100组目标训练数据\n冻结CNN回归头',886,310,320,102,27);
 text(s,'区间宽且重叠，不能据最高点宣称显著优势。',886,477,320,133,26,false,muted);
}
// 13
{
 const s=slide('负结果与当前限制','M5_FINAL_REPORT.md 研究问题与限制');
 text(s,'冻结DNA表征及融合没有带来性能增益。',75,164,1125,70,33,true);
 text(s,'当前实验只检验固定编码器和预定回归头，不代表所有预训练方案。',75,247,1125,75,27);
 text(s,'数据和评价条件限制泛化结论。',75,364,1125,65,33,true);
 text(s,'天然表达标签包含序列之外因素，教师数据元信息缺失。\nval参与模型选择，历史随机划分可能共享邻近上下文。',75,448,1125,136,28);
}
// 14
{
 const s=slide('RQ1至RQ3的回答','evaluation/；explanation_analysis/；mutation/');
 const items=[['RQ1','CNN在当前验证基准上领先，解释度仍有限。热力学模型可运行，预测天然表达较弱。'],['RQ2','−10得到两方法的描述性富集支持，具体Top-k随方法变化。'],['RQ3','高归因位点更敏感，一阶方向高度自洽。跨模型一致性弱，生物因果仍需实测。']];
 for(let i=0;i<items.length;i++){text(s,items[i][0],75,173+i*151,145,65,35,true,blue);text(s,items[i][1],238,174+i*151,957,115,29);}
}
// 15
{
 const s=slide('复现与最终交付','configs/m4_protocol_v1.json；verification.json');
 text(s,'统一入口',75,167,1120,55,33,true);
 text(s,'python -m pr02 m4 --stage verify',75,242,1120,63,31,false,blue);
 text(s,'代码与结果',75,349,1120,55,33,true);
 text(s,'固定配置、输入与源码摘要、逐条预测、特征ID和可编辑图表。',75,425,1120,67,28);
 text(s,'下一步研究',75,533,1120,55,33,true);
 text(s,'优先取得独立实验数据，开展严格基因组分组评价和真实突变测量。',75,599,1120,54,25,false,muted);
}
// 16
{
 const s=slide('数据与方法来源','仓库来源清单及最终报告参考文献');
 const refs=[['原始50bp数据','HaochenW/Deep_promoter 与原论文'],['TSS定位证据','Thomason等TSS图谱及历史参考U00096.2'],['热力学模型','Promoter Calculator，Nature Communications 2022'],['DNA预训练模型','DNABERT-2，官方117M固定revision'],['本项目结果','Azusakada/PR02-01-promoter--insight']];
 table(s,['来源','用途'],refs,{height:390,columnWidths:[330,806]});
 text(s,'教师CSV的论文、单位、TSS和测量条件仍记为unknown。',75,588,1120,52,25,false,muted);
 s.speakerNotes.text=content.references.map(r=>r.join(': ')).join('\n');
}

const candidate=path.join(temp,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=process.env.PR02_PPTX_OUTPUT||path.join(repo,'reports/M5_DEFENSE_FINAL.pptx');
const result=await finalizePresentation({workspaceDir:repo,candidatePath:candidate,finalPath,
 pythonExecutable:process.env.PR02_RUNTIME_PYTHON,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--cover-role','cover','--validate-bullet-geometry','--validate-heading-fit',
  '--require-native-table-slide','3','--require-native-table-slide','8','--require-native-table-slide','16'],
 explicitTotalSlideCount:16,requiredNativeTableOwnerSlides:[3,8,16],requiredNativeChartOwnerSlides:[4,5,6,7,9,10,11,12],
 fontPolicy:{basis:'design',families:[font]},materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,
 receiptPath:path.join(temp,'validation-'+path.basename(finalPath)+'.json')});
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(temp,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log(JSON.stringify({finalPath,result}));
