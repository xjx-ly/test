import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.charts import Pie
from pyecharts.charts import Line as Line1
from pyecharts.faker import Faker
from pyecharts.charts import Grid
from pyecharts.charts import Scatter as Scatter1

from streamlit_echarts import st_pyecharts




from statsmodels.graphics.tsaplots import plot_acf,plot_pacf  #自相关图、偏自相关图



from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm_notebook
from statsmodels.tsa.arima_model import ARIMA
import warnings
# 忽视在模型拟合中遇到的错误
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox


import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.tools.eval_measures import mse
from statsmodels.tools.eval_measures import meanabs
from scipy.stats import pearsonr

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

from pathlib import Path

path1 = Path(__file__).parents[1] / 'Streamlit/1.csv'
path2 = Path(__file__).parents[1] / 'Streamlit/2.csv'
path3 = Path(__file__).parents[1] / 'Streamlit/4.xlsx'


def Layouts_plotly():
    st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    )

    
    st.sidebar.write('导航栏')


    biao =st.sidebar.selectbox('',['🏠主页',''])
    
    if biao=="🏠主页":
        st.markdown('# 🏠 基于商场销售数据的研究与分析')
  
        
        
    
    
    biao =st.sidebar.selectbox('',['表格数据展示','原始数据','处理后的数据',])

    with st.container():
        if biao=='原始数据':
            st.markdown('#### 商场销售数据可视化')
            Double_coordinates()
        elif biao=='处理后的数据':
            Double_coordinates2()
            
    tu =st.sidebar.selectbox('',['图展示 ','数据预处理','营业额','利润率','打折力度'])
    with st.container():
        df,df2=ChuLi()
        x=df2['create_dt']
        Yye=df2['Yye']
        LrL=df2['LrL']
        R=df2['R']
        if tu=='营业额':
            line('营业额',x,Yye,'营业额/万元')

        elif tu=='利润率':
            line('利润率',x,LrL,'利润率/%')
            
        elif tu=='打折力度':
            line('打折力度',x,R,'打折力度')
        elif tu=='数据预处理':
            pie(1)
            pie(2)
            
            
            


 #   时序预测
    sr =st.sidebar.selectbox('',['预测 ','SARIMA预测'])

    with st.container():
        if sr=='SARIMA预测':
            col1, col2 = st.columns((4, 1)) 
            tab1, tab2, tab3 = col1.tabs(["图表", "模型检验", "预测"])
            with tab1:
                st.markdown('#### BIC数据展示')
                sar("sar1")
            with tab2:
                st.markdown('#### 模型检验')
                sar("sar2")
            with tab3:
                st.markdown('#### 预测')
                sar("sar3")   
            
            

                
#   关系探究         
    option =st.sidebar.selectbox('',['关系探究','营业额图','打折力度与销售额','打折力度与利润率'])
    with st.container():
        df,df2=ChuLi()
        Y_17=df2[(df2['create_dt']<"2018-01-01") & (df2['create_dt']>="2017-01-01") ]
        Y_18=df2[(df2['create_dt']<"2019-01-01") & (df2['create_dt']>="2018-01-01") ]
        if option=='打折力度与销售额':
            sort='Yye'
            year(option,sort,'销售额/万元')
        if option=='打折力度与利润率':
            sort='LrL'
            year(option,sort,'利润率/%')
            
        if option == '营业额图':
            st.markdown('#### 2017年及2018年每天的营业额图')
            Y_17_18(Y_17,Y_18)
            
            
#    大区分类         
    option =st.sidebar.selectbox('',['大区分类','商品类别展示'])
    with st.container():

        if option=='商品类别展示':
            st.markdown('### 商品类别展示')
            option =st.selectbox('',[' ','一级商品类目','二级商品类目','三级商品类目'])
            with st.container():
                if option == "一级商品类目":
                    bar()
                    Gx('first_category_id','first_category_name')
                if option =="二级商品类目":
                    Gx('second_category_id','second_category_name')
                if option =="三级商品类目":
                    Gx('third_category_id','third_category_name')                
                    
                    
                
                
def Gx(category_id, category_name):
    d4 = load_data2()
    sort = (d4.groupby([category_id, category_name],
                       as_index=False,
                       sort=False)['sku_id'].count()).sort_values(
                           'sku_id', ascending=False)

    st.markdown('##### 商品类目表格')
    st.dataframe(sort.reset_index(drop=True))  #  索引从0开始
    name = sort[category_name].tolist()
    st.markdown('#### 类别')
    lb = st.selectbox('', name)
    with st.container():
        first_category_id = None
        for index, friend in enumerate(name):
            if friend == lb:
                y_id = sort[sort[category_name] == lb].iloc[0, 0]
                category_sku_id = d4.groupby(category_id).get_group(y_id)

                col1, col2 = st.columns((3, 1))
                tab1, tab2, = col1.tabs([
                    "数据",
                    "模型",
                ])
                with tab1:
                    st.markdown('##### ' + friend + '数据展示')
                    dq(category_sku_id, lb, 1)
                with tab2:
                    dq(category_sku_id, lb, 2)
 
        
            
            
def year(option, sort, name):
    df, df1 = ChuLi()
    df_10 = df1[df1['Yye'] < 10].sort_values('Yye', ascending=False)
    hg = st.selectbox('年份选择', [' ', '2017年', '2018年'])
    with st.container():
        time1 = None
        time2 = None
        t1 = hg
        t2 = ""
        if hg == "2017年":
            time1 = "2017"
            time2 = "2018"
            genre = st.radio("", ('2017年数据', '上下半年数据', '季度数据'))
            if genre == '上下半年数据':
                hg1 = st.selectbox('上下半年份选择', ['2017上半年', '2017下半年'])
                with st.container():
                    t2 = hg1
                    if hg1 == "2017上半年":
                        time2 = "2017-06-01"
                    if hg1 == "2017下半年":
                        time2 = "2018"
            elif genre == '季度数据':
                time2 = "2017-04-01"

                hg2 = st.selectbox('季度选择', ['一季度 ', '二季度', '三季度', '四季度'])

                with st.container():
                    t2 = hg2
                    if hg2 == "一季度":
                        time1 = "2017-01-01"
                        time2 = "2017-04-01"

                    if hg2 == "二季度":
                        time1 = "2017-04-01"
                        time2 = "2017-07-01"

                    if hg2 == "三季度":
                        time1 = "2017-07-01"
                        time2 = "2017-10-01"
                    if hg2 == "四季度":
                        time1 = "2017-10-01"
                        time2 = "2018"

        if hg == "2018年":
            time1 = "2018"
            time2 = "2019"
            genre2 = st.radio("", ('2018年数据', '上下半年数据', '季度数据'))
            if genre2 == '上下半年数据':
                hg1 = st.selectbox('上下半年份选择', [' ', '2018上半年', '2018下半年'])
                with st.container():
                    t1 = hg1
                    if hg1 == "2018上半年":
                        time2 = "2018-06-01"
                    if hg1 == "2018下半年":
                        time2 = "2019"
            elif genre2 == '季度数据':
                time2 = "2018-04-01"
                hg2 = st.selectbox('季度选择', [' ', '一季度 ', '二季度', '三季度', '四季度'])
                with st.container():
                    t2 = hg2
                    if hg2 == "一季度":
                        time1 = "2018"
                        time2 = "2018-04-01"
                    if hg2 == "二季度":
                        time1 = "2018-04-01"
                        time2 = "2018-07-01"
                    if hg2 == "三季度":
                        time1 = "2018-07-01"
                        time2 = "2018-10-01"
                    if hg2 == "四季度":
                        time1 = "2018-10-01"
                        time2 = "2019"

        if time1 != None:
            df = df_10[(df_10['create_dt'] >= time1)
                       & (df_10['create_dt'] < time2)]
            xuanze(option, t1, t2, df, sort, name)
            
               
def xuanze(option,t1,t2,df,sort,name):
    col1, col2 = st.columns((3, 1)) 
    tab1, tab2,= col1.tabs(["数据", "模型"])
                
    with tab2:
        st.markdown('#### '+t1+t2+option)
        Huigui(df,df[sort],name)
    with tab1:
        st.markdown('#### 数据展示')
        st.dataframe(df)
        


def Bubble():       
    df = px.data.gapminder()
    
    fig = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp",
              size="pop", color="continent",
                     hover_name="country", log_x=True, size_max=60)
    # Plot the data
    st.plotly_chart(fig)

    
def Scatter():
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    
    # Plot the data
    st.plotly_chart(fig)    

    
def Line():
    df = px.data.stocks()
    fig = px.line(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    st.plotly_chart(fig)    
    
    
    
    
@st.cache(allow_output_mutation=True)
def load_data():
    d1 =pd.read_csv(path1,encoding="utf-8")
    d2 =pd.read_csv(path2,encoding="utf-8")
    df=pd.concat([d1,d2])   #合并数据
    df = df.reset_index(drop=True)

    return df


@st.cache(allow_output_mutation=True)
def ChuLi():
    data_c = load_data()

    data_1 = data_c[data_c['is_finished'] == 1]
    data_1.loc[(data_1['create_dt'] >= '2019-01-02') &
               (data_1['sku_id'] == 2003118305), 'sku_sale_prc'] = data_1.loc[
                   (data_1['create_dt'] >= '2019-01-02') &
                   (data_1['sku_id'] == 2003118305), 'sku_prc']

    data_1.loc[(data_1['create_dt'] >= '2018-11-01') &
               (data_1['sku_id'] == 2013509486), 'sku_sale_prc'] = data_1.loc[
                   (data_1['create_dt'] >= '2018-11-01') &
                   (data_1['sku_id'] == 2013509486), 'sku_prc']

    data_1.loc[(data_1['create_dt'] >= '2018-12-29') &
               (data_1['sku_id'] == 2016591037), 'sku_sale_prc'] = data_1.loc[
                   (data_1['create_dt'] >= '2018-12-29') &
                   (data_1['sku_id'] == 2016591037), 'sku_prc']

    # 没有成本  成本为30%利润率

    data_1[(data_1['sku_cost_prc'] == 0)
           & (data_1['sku_prc'] == data_1['sku_sale_prc'])]

    data_1.loc[(data_1['sku_cost_prc'] == 0) &
               (data_1['sku_prc'] == data_1['sku_sale_prc']),
               'sku_cost_prc'] = data_1.loc[
                   (data_1['sku_cost_prc'] == 0) &
                   (data_1['sku_prc'] == data_1['sku_sale_prc']),
                   'sku_sale_prc'] * 0.7

    #  降价没有成本  成本价应为降价后的销售价
    data_1[(data_1['sku_cost_prc'] == 0)
           & (data_1['sku_prc'] != data_1['sku_sale_prc'])]

    data_1.loc[(data_1['sku_cost_prc'] == 0) &
               (data_1['sku_prc'] != data_1['sku_sale_prc']),
               'sku_cost_prc'] = data_1.loc[(data_1['sku_cost_prc'] == 0) & (
                   data_1['sku_prc'] != data_1['sku_sale_prc']),
                                            'sku_sale_prc']

    #总收入
    data_1['Yye'] = data_1['sku_cnt'] * data_1['sku_sale_prc']

    #总成本
    data_1['Zcb'] = data_1['sku_cnt'] * data_1['sku_cost_prc']

    data_Yye = data_1.groupby('create_dt',
                              as_index=False)['Yye', 'Zcb', 'sku_prc',
                                              'sku_sale_prc'].sum()
    data_Yye['Yye'] = data_Yye['Yye'] * 0.0001
    data_Yye['Zcb'] = data_Yye['Zcb'] * 0.0001

    #利润
    data_Yye['Lr'] = (data_Yye['Yye'] - data_Yye['Zcb']).round(2)

    #利润率
    data_Yye['LrL'] = (data_Yye['Lr'] / data_Yye['Zcb'] * 100).round(2)

    # 打折力度
    data_Yye['R'] = ((1 - data_Yye['sku_sale_prc'] / data_Yye['sku_prc']) *
                     100).round(2)

    return data_1, data_Yye



def Double_coordinates():  
    df = load_data()
    
    st.markdown('#### 基础数据表展示')
    st.dataframe(df)   
    
def Double_coordinates2():
    df1,df2 = ChuLi()

    st.markdown('#### 处理后的数据展示')
    st.dataframe(df1) 
    st.markdown('#### 处理后的数据展示')
    st.dataframe(df2)
    
def pie(col,):
    
    df = load_data()

    if col==1:
        st.markdown('#### 订单统计')
        row_num1=len(df[df['is_finished']==1])
        row_num0=len(df[df['is_finished']==0])
        t=["有效订单","无效订单"]
        t1=[row_num1,row_num0]
        color=["Orange", "#f33165"]
    elif col==2:
        st.markdown('#### 有效订单')
        #  sku_sale_prc为0
        s1=len(df[df['sku_cost_prc']==0])
        
        # 成本未缺失
        s2=len(df[df['sku_cost_prc']>0])
        
        #  降价没有成本  成本价应为降价后的销售价
        s3=len(df[(df['sku_cost_prc']==0) & (df['sku_prc']!=df['sku_sale_prc'])])           
        t=["成本价缺失","降价后成本价缺失",'成本价未缺失']
        t1=[s1,s3,s2]
        color=["Orange","green", "#f33165"]
    
    pie=Pie()
    pie.add("", [list(z) for z in zip(t, t1)],is_clockwise = False)
    pie.set_colors(color)
    pie.set_global_opts(title_opts=opts.TitleOpts())
    pie.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))

    st_pyecharts( pie)

    
def line(name, x, y, y_table):

    st.markdown('#### 每天的' + name)

    line = Line1(init_opts=opts.InitOpts(
        width="5800px",
        height="300px",
    ))
    line.add_xaxis(x.tolist())
    line.add_yaxis(name,
                   y.tolist(),
                   markline_opts=opts.MarkLineOpts(
                       data=[
                           opts.MarkLineItem(type_="average", name="平均值"),
                           opts.MarkLineItem(type_="min", name="最小点"),
                           opts.MarkLineItem(symbol="circle",
                                             type_="max",
                                             name="最高点"),
                       ],
                       linestyle_opts=opts.LineStyleOpts(
                           type_="dashed",
                           color='red',
                       )))
    # line.set_global_opts(titl00e_opts=opts.TitleOpts(title="每天利润率"))
    line.set_global_opts(
        yaxis_opts=opts.AxisOpts(
            name=y_table,
            name_gap=40,
            name_location='middle',
            name_textstyle_opts=opts.TextStyleOpts(font_size=14)),
        xaxis_opts=opts.AxisOpts(name='年份',
                                 name_gap=40,
                                 name_rotate=60,
                                 axislabel_opts={"rotate": 45}),
        legend_opts=opts.LegendOpts(
            type_=None  # 'plain'：普通图例。缺省就是普通图例。 
            # 'scroll'：可滚动翻页的图例。当图例数量较多时可以使用。
            ,
            pos_left='right'  #图例横向的位置,right表示在右侧，也可以为百分比
            ,
            pos_top='middle'  #图例纵向的位置，middle表示中间，也可以为百分比
        ))
    line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    st_pyecharts(line)
    
    

#  预测
# 找最优的参数 SARIMAX
@st.cache
def find_best_params(data: np.array, ):
    # ARIMA的参数
    ps = range(0, 3)
    d = range(0, 2)
    qs = range(0, 3)
    # 季节项相关的参数
    Ps = range(0, 3)
    D = range(0, 2)
    Qs = range(0, 3)
    # 将参数打包，传入下面的数据，是哦那个BIC准则进行参数选择
    params_list = list(product(ps, d, qs, Ps, D, Qs))
    print(params_list)

    result = []
    best_bic = 100000
    for param in tqdm_notebook(params_list):

        # 模型拟合

        model = SARIMAX(data,
                        order=(param[0], param[1], param[2]),
                        seasonal_order=(param[3], param[4], param[5],
                                        7)).fit(disp=-1)
        bicc = model.bic  # 拟合出模型的BIC值
        # print(bic)
        # 寻找最优的参数
        if bicc < best_bic:
            best_mode = model
            best_bic = bicc
            best_param = param
        param_1 = (param[0], param[1], param[2])
        param_2 = (param[3], param[4], param[5], 7)
        param = 'SARIMA{0}x{1}'.format(param_1, param_2)
        print(param)
        result.append([param, model.bic])

    result_table = pd.DataFrame(result)
    result_table.columns = ['parameters', 'bic']
    result_table = result_table.sort_values(
        by='bic', ascending=True).reset_index(drop=True)
    print(result_table)
    return result_table
    
#  预测
def sar(table):
    df1, df2 = ChuLi()
    Yye = df2[(df2['create_dt'] < "2019-01-01")
              & (df2['create_dt'] >= "2017-01-01")]
    result_table = find_best_params(Yye['Yye'].astype(float))
    ma1 = SARIMAX(Yye["Yye"].astype(float),
                  order=(2, 1, 1),
                  seasonal_order=(0, 1, 2, 7)).fit(disp=-1)
    resid = ma1.resid

    n_steps = len(Yye) + 1
    f = ma1.predict(1, n_steps)  # 95% CI
    idx = pd.date_range('2017-01-01', periods=n_steps, freq='D')
    fc = pd.DataFrame(np.column_stack([f]), index=idx, columns=['predict'])

    # 赋值是按照索引来赋值的，   索引不同 后加tolist（）
    data_indexf = fc.reset_index()
    data_indexf.columns = ['create_dt', 'predict']

    data_indexf.loc[data_indexf['create_dt'] > '2018-12-24',
                    'predict'] = data_indexf.loc[
                        data_indexf['create_dt'] >= '2018-12-24',
                        'predict'][:-1].tolist()

    data_24 = data_indexf.drop(
        data_indexf[data_indexf['create_dt'] == '2018-12-24'].index)
    data_24['Yye'] = Yye.Yye.tolist()
    data_24['create_dt'] = data_24['create_dt'].dt.date

    if table == "sar1":
        st.dataframe(result_table)
        line("销售额", Yye['create_dt'], Yye["Yye"], "销售额/万元")
        st.markdown('#### 原始预测数据展示')
        st.dataframe(data_24)
        syuc(data_24)

    # 检验
    if table == "sar2":
        fig = ma1.plot_diagnostics(figsize=(10, 8))
        st.pyplot(fig)

        st.text(ma1.summary().as_text())

        st.text("==============================")
        st.text('D-W检验的结果为：' +
                str(sm.stats.durbin_watson(resid.values).round(2)))
        st.text('残差序列的白噪声检验结果为： \n ' +
                str(acorr_ljungbox(resid, lags=1).round(2)))  #返回统计量、P值

        predict_10 = data_24.Yye
        fact_10 = data_24.predict
        st.text('MSE：{:.2f}'.format(mean_squared_error(predict_10,
                                                       fact_10)))  # MSE 均方误差
        st.text('NAE：{:.2f}'.format(mean_absolute_error(
            predict_10, fact_10)))  # MAE 平均绝对误差
        st.text('RMSE：{:.2f}'.format(
            np.sqrt(mean_squared_error(predict_10, fact_10))))
        st.text('R square：{:.2f}'.format(r2_score(predict_10, fact_10)))

    n_steps = 31
    f = ma1.forecast(steps=n_steps)  # 95% CI
    idx = pd.date_range('2019-01-01', periods=n_steps, freq='D')
    fc1 = pd.DataFrame(np.column_stack([f]), index=idx, columns=['forecast'])

    data_24_index = data_24.set_index('create_dt')
    data_forecast = pd.concat((data_24_index, fc1.forecast), axis=1)
    data_forecast.columns = [
        'predict',
        'Yye',
        'forecast',
    ]

    data_reset_index = data_forecast.reset_index()
    data_reset_index.columns = [
        'create_dt',
        'predict',
        'Yye',
        'forecast',
    ]

    if table == "sar3":
        
        yc(data_reset_index)
        st.markdown('##### 局部图')
        data_18=data_reset_index[data_reset_index['create_dt']>='2018-06-01']
        yc(data_18)
        
        
def yc(data_reset_index):
    line = Line1()
    line.add_xaxis(
        data_reset_index.create_dt.dt.date.tolist())  #.tolist() 转换成list
    line.add_yaxis("原始数据",
                   data_reset_index.Yye.tolist(),
                   is_connect_nones=True,
                   color="blue")
    line.add_yaxis("预测数据", data_reset_index.predict.tolist(), color="red")
    line.add_yaxis("预测31天", data_reset_index.forecast.tolist(), color="green")

    # line.set_global_opts(titl00e_opts=opts.TitleOpts(title="每天利润率"))
    line.set_global_opts(
        yaxis_opts=opts.AxisOpts(
            name='销售额',
            name_gap=40,
            name_location='middle',
            name_textstyle_opts=opts.TextStyleOpts(font_size=14)),
        xaxis_opts=opts.AxisOpts(name='年份',
                                 name_gap=40,
                                 name_rotate=60,
                                 axislabel_opts={"rotate": 45}),
        legend_opts=opts.LegendOpts(
            type_=None  # 'scroll'：普通图例。缺省就是普通图例。 
            # 'scroll'：可滚动翻页的图例。当图例数量较多时可以使用。
            ,
            pos_left='65%'  #图例横向的位置,right表示在右侧，也可以为百分比
            ,
            pos_top='8%'  #图例纵向的位置，middle表示中间，也可以为百分比
        ))
    line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    st_pyecharts(line)

    
    
    
def syuc(data_24):
    line = Line1()
    line.add_xaxis(data_24.create_dt.tolist())  #.tolist() 转换成list
    line.add_yaxis("原始数据",
                   data_24.Yye.tolist(),
                   is_connect_nones=True,
                   color="blue")
    line.add_yaxis("预测数据", data_24.predict.tolist(), color="red")

    # line.set_global_opts(titl00e_opts=opts.TitleOpts(title="每天利润率"))
    line.set_global_opts(
        yaxis_opts=opts.AxisOpts(
            name='销售额',
            name_gap=40,
            name_location='middle',
            name_textstyle_opts=opts.TextStyleOpts(font_size=14)),
        xaxis_opts=opts.AxisOpts(name='年份',
                                 name_gap=40,
                                 name_rotate=60,
                                 axislabel_opts={"rotate": 45}),
        legend_opts=opts.LegendOpts(
            type_=None  # 'scroll'：普通图例。缺省就是普通图例。 
            # 'scroll'：可滚动翻页的图例。当图例数量较多时可以使用。
            ,
            pos_left='75%'  #图例横向的位置,right表示在右侧，也可以为百分比
            ,
            pos_top='8%'  #图例纵向的位置，middle表示中间，也可以为百分比
        ))
    line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    st_pyecharts(line)
    
    
# 关系探究
def Huigui(time, sort, name):

    x = np.array(time['R'].astype('float64'))  #一元线性回归分析
    y = np.array(sort.astype('float64'))
    X = sm.add_constant(x)  #向x左侧添加截距列x0=[1,……,1]
    model = sm.OLS(endog=y, exog=X)  #建立最小二乘估计
    fit = model.fit()  #拟合模型

    y1 = fit.fittedvalues

    jg(fit, x, y, y1)

    line = (
        Line1().add_xaxis(x).add_yaxis(
            "test", y1, is_symbol_show=False, is_smooth=True,
            color="red").set_global_opts(
                yaxis_opts=opts.AxisOpts(
                    name='销售额/万元',
                    name_gap=40,
                    name_location='middle',
                    name_textstyle_opts=opts.TextStyleOpts(font_size=18)),
                xaxis_opts=opts.AxisOpts(
                    name='打折力度/%',
                    name_location='middle'  #坐标轴名字所在的位置
                    ,
                    name_textstyle_opts=opts.TextStyleOpts(font_size=18))))
    scatter = (Scatter1().add_xaxis(x).add_yaxis("data", y))
    line.overlap(scatter)
    line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    st_pyecharts(line, height="500px")

    if fit.pvalues[0] >= 0.05:

        agree = st.checkbox('模型改进')
        if agree:
            #             st.write('Great!')
            huigui2(time, sort, name)
    

    
    
    
    
    
    
    
def jg(fit,x,y,y1):
    st.text(fit.summary().as_text())
    st.text("=====================================")
#     st.text("Parmeters:"+str( fit.params))
    st.text("rsquared: "+str(fit.rsquared.round(2)))
    pc = pearsonr(x,y)
    st.text("相关系数："+str(pc[0].round(2)))
    st.text("MSE均方误差:"+str(mse(y,y1).round(2)))
    st.text("MAE平均绝对误差:"+str(meanabs(y,y1).round(2)))
    st.text("=====================================")    
    
    

            
            
def huigui2(time, sort, name):

    x = np.array(time['R'].astype('float64'))  #一元线性回归分析
    y = np.array(sort.astype('float64'))
    #     X=sm.add_constant(x)#向x左侧添加截距列x0=[1,……,1]
    model = sm.OLS(endog=y, exog=x)  #建立最小二乘估计
    fit = model.fit()  #拟合模型

    y1 = fit.fittedvalues

    line = (
        Line1().add_xaxis(x).add_yaxis(
            "test", y1, is_symbol_show=False, is_smooth=True,
            color="red").set_global_opts(title_opts=opts.TitleOpts(
                title="Overlap-line+scatter")).set_global_opts(
                    yaxis_opts=opts.AxisOpts(
                        name='销售额/万元',
                        name_gap=40,
                        name_location='middle',
                        name_textstyle_opts=opts.TextStyleOpts(font_size=18)),
                    xaxis_opts=opts.AxisOpts(
                        name='打折力度/%',
                        name_location='middle'  #坐标轴名字所在的位置
                        ,
                        name_textstyle_opts=opts.TextStyleOpts(font_size=18))))
    scatter = (Scatter1().add_xaxis(x).add_yaxis("data", y))
    line.overlap(scatter)
    line.set_global_opts(xaxis_opts=opts.AxisOpts(name='X轴名称'))
    line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    st_pyecharts(line, height="500px")

    jg(fit, x, y, y1)
    

def Y_17_18(df1, df2):
    line1 = Line1()
    line1.add_xaxis(df1.create_dt.tolist())  #.tolist() 转换成list
    line1.add_yaxis("营业额",
                    df1.Yye.tolist(),
                    is_connect_nones=False,
                    markline_opts=opts.MarkLineOpts(
                        data=[
                            opts.MarkLineItem(type_="average", name="平均值"),
                            opts.MarkLineItem(type_="min", name="最小点"),
                            opts.MarkLineItem(symbol="circle",
                                              type_="max",
                                              name="最高点"),
                        ],
                        linestyle_opts=opts.LineStyleOpts(
                            type_="dashed",
                            color='red',
                        )))

    #line.set_global_opts(
    line1.set_global_opts(
        yaxis_opts=opts.AxisOpts(
            name='营业额/万元',
            name_location='middle',
            name_gap=25,
            name_textstyle_opts=opts.TextStyleOpts(font_size=14)),
        xaxis_opts=opts.AxisOpts(name='年份',
                                 name_rotate=60,
                                 name_gap=40,
                                 axislabel_opts={"rotate": 45}),
        legend_opts=opts.LegendOpts(
            type_=None  # 'plain'：普通图例。缺省就是普通图例。 
            # 'scroll'：可滚动翻页的图例。当图例数量较多时可以使用。
            ,
            pos_left='right'  #图例横向的位置,right表示在右侧，也可以为百分比
            ,
            pos_top='middle'  #图例纵向的位置，middle表示中间，也可以为百分比
        ))
    line1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

    line2 = Line1()
    line2.add_xaxis(df2.create_dt.tolist())  #.tolist() 转换成list
    line2.add_yaxis("营业额",
                    df2.Yye.tolist(),
                    is_connect_nones=False,
                    markline_opts=opts.MarkLineOpts(
                        data=[
                            opts.MarkLineItem(
                                type_="average",
                                name="平均值",
                            ),
                            opts.MarkLineItem(type_="min", name="最小点"),
                            opts.MarkLineItem(symbol="circle",
                                              type_="max",
                                              name="最高点"),
                        ],
                        linestyle_opts=opts.LineStyleOpts(
                            type_="dashed",
                            color='red',
                        )))

    #line.set_global_opts()
    line2.set_global_opts(
        yaxis_opts=opts.AxisOpts(
            name='营业额/万元',
            name_location='middle',
            name_gap=25,
            name_textstyle_opts=opts.TextStyleOpts(font_size=14)),
        xaxis_opts=opts.AxisOpts(name='年份',
                                 name_rotate=60,
                                 name_gap=40,
                                 axislabel_opts={"rotate": 45}),
        legend_opts=opts.LegendOpts(
            type_=None  # 'plain'：普通图例。缺省就是普通图例。 
            # 'scroll'：可滚动翻页的图例。当图例数量较多时可以使用。
            ,
            pos_left='right'  #图例横向的位置,right表示在右侧，也可以为百分比
            ,
            pos_top='middle'  #图例纵向的位置，middle表示中间，也可以为百分比
        ))
    line2.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

    grid = (Grid().add(line1,
                       grid_opts=opts.GridOpts(pos_bottom="60%",
                                               pos_right="10%")).add(
                                                   line2,
                                                   grid_opts=opts.GridOpts(
                                                       pos_top="60%",
                                                       pos_left="10%")))
    st_pyecharts(grid, height="600px")
    
    

    

# 大区分类
@st.cache(allow_output_mutation=True)
def load_data2():
    df = pd.read_excel(path3)

    return df


def bar():
    df = load_data2()

    S = (df.groupby('first_category_name', as_index=False,
                    sort=False)['first_category_id'].count()).sort_values(
                        'first_category_id', ascending=False)
    c = (
        Bar(init_opts=opts.InitOpts(
            width="800px",
            height="400px",
        )).add_xaxis(S.first_category_name.tolist()).add_yaxis(
            '商品', S.first_category_id.tolist()).set_global_opts(
                yaxis_opts=opts.AxisOpts(
                    name='商品数量',
                    name_gap=40,
                    name_location='middle',
                    name_textstyle_opts=opts.TextStyleOpts(font_size=14)),
                xaxis_opts=opts.AxisOpts(name='种类',
                                         name_gap=40,
                                         name_rotate=60,
                                         axislabel_opts={"rotate": 45}),
                legend_opts=opts.LegendOpts(
                    type_=None  # 'plain'：普通图例。缺省就是普通图例。 
                    # 'scroll'：可滚动翻页的图例。当图例数量较多时可以使用。
                    ,
                    pos_left='right'  #图例横向的位置,right表示在右侧，也可以为百分比
                    ,
                    pos_top='middle'  #图例纵向的位置，middle表示中间，也可以为百分比
                )).
        set_series_opts(label_opts=opts.LabelOpts(
            position=
            'top'  #设置数据标签所在的位置 'top'，'left'，'right'，'bottom'，'inside'，'insideLeft'，'insideRight'
            # 'insideTop'，'insideBottom'， 'insideTopLeft'，'insideBottomLeft'
            # 'insideTopRight'，'insideBottomRight'
            ,
            color='Black'  #数据标签的颜色
            ,
            font_size=12
            # ,formatter #数据标签显示格式
        )  ##设置数据标签的格式s
                        ))
    st_pyecharts(c, )

    
    
    
def dq(category_sku_id,option,table):
    df1,df2 = ChuLi()
    
    #粮油副食
    t1=pd.merge(left=category_sku_id,right=df1,on="sku_id")    
    #每天营业额，成本
    L=t1.groupby('create_dt',as_index=False)[
        'sku_prc','sku_sale_prc','Yye','Zcb'].sum()

    #利润
    L['Lr']=(L['Yye']-L['Zcb']).round(2)

    #利润率
    L['LrL']=(L['Lr']/L['Zcb']*100).round(2)

    L['Yye']=L['Yye']*0.0001

    #打折力度
    L['R']=((1-L['sku_sale_prc']/L['sku_prc'])*100).round(2)
    
    if table == 1:
        st.dataframe(L)
        line("销售额",L['create_dt'],L['Yye'],'销售额/万元')
        line("利润率",L['create_dt'],L['LrL'],'利润率/%')
        line("打折力度",L['create_dt'],L['R'],'打折力度/%')
    
    if table == 2:
        
        st.markdown('##### '+option+"打折力度与销售额")
        Huigui(L,L['Yye'],'销售额/万元')

        st.markdown('##### '+option+"打折力度与利润率")
        Huigui(L,L['LrL'],'利润率/%')
    

    
# 分析
    
def line_scatter():
    x = Faker.choose()
    line = (
        Line1()
        .add_xaxis(x)
        .add_yaxis("商家A", Faker.values())
        .set_global_opts(title_opts=opts.TitleOpts(title="Overlap-line+scatter"))
    )
    scatter = (
        Scatter1()
        .add_xaxis(x)
        .add_yaxis("商家A", Faker.values())
    )
    line.overlap(scatter)
    st_pyecharts(line)   
    
    

    
def main():
    Layouts_plotly()


if __name__ == "__main__":
    main()
    
    
    
    

    
    
  
