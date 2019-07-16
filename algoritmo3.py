import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data.psychsignal import twitter_withretweets as twitter_sentiment
from quantopian.pipeline.data.morningstar import Fundamentals
from quantopian.pipeline.data import EquityPricing
#from quantopian.pipeline.data import Fundamentals

MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 600

MAX_SHORT_POSITION_SIZE = 2.0 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 2.0 / TOTAL_POSITIONS

def initialize(context):
    
    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')
    
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)

    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)


def make_pipeline():
    
    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest
    
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=3,
    )
    
    total_revenue = Fundamentals.total_revenue.latest
    
    yesterday_close = EquityPricing.close.latest
    yesterday_volume = EquityPricing.volume.latest
    
    test_sentiment = (
        twitter_sentiment.bull_scored_messages.latest/
        twitter_sentiment.total_scanned_messages.latest
    )

    universe = QTradableStocksUS()
    
    value_winsorized = value.winsorize(min_percentile=0.05, max_percentile=0.95)
    quality_winsorized = quality.winsorize(min_percentile=0.05, max_percentile=0.95)
    sentiment_score_winsorized = sentiment_score.winsorize(min_percentile=0.05,                                                                             max_percentile=0.95)
    close_winsorized = yesterday_close.winsorize(min_percentile=0.05,     max_percentile=0.95)
    volume_winsorized = yesterday_volume.winsorize(min_percentile=0.05, max_percentile=0.95)

    combined_factor = (
        value_winsorized.zscore() + 
        quality_winsorized.zscore() + 
        ((sentiment_score_winsorized.zscore()+test_sentiment.zscore())/2)+
        total_revenue.zscore()+
        close_winsorized.zscore()+
        volume_winsorized.zscore()   
    )
    
    longs = combined_factor.top(TOTAL_POSITIONS//2, mask=universe)
    shorts = combined_factor.bottom(TOTAL_POSITIONS//2, mask=universe)

    long_short_screen = (longs | shorts)

    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'combined_factor': combined_factor,
            'total_revenue': total_revenue,
            'close': yesterday_close,
            'volume': yesterday_volume,            
        },
        screen=long_short_screen
    )
    return pipe


def before_trading_start(context, data):
    
    context.pipeline_data = algo.pipeline_output('long_short_equity_template')

    context.risk_loadings = algo.pipeline_output('risk_factors')


def record_vars(context, data):
    
    algo.record(num_positions=len(context.portfolio.positions))

def rebalance(context, data):

    pipeline_data = context.pipeline_data

    risk_loadings = context.risk_loadings
    
    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)

    constraints = []
    
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))

    constraints.append(opt.DollarNeutral())

    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)
    
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )
