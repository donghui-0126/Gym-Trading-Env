class Portfolio:
    def __init__(self, asset, fiat, init_value):
        self.asset =asset
        self.fiat =fiat
        self.init_value = init_value
                
    def valorisation(self, price):
        return sum([
            self.asset * price,
            self.fiat,
        ])
        
    def position(self, price):
        return self.asset * price / self.init_value
    
    def trade_to_position(self, target_position, price, trading_fees, precious_position,init_value):
        # 진입하는 포지션
        transaction_fiat = (target_position - precious_position) * init_value   

        # long trade
        if transaction_fiat > 0:
            asset_fiat = -transaction_fiat
            asset_trade = transaction_fiat / price
            
            self.fiat = self.fiat + asset_fiat
            self.asset = self.asset + asset_trade * (1-trading_fees)
            
        # shrot trade
        else:
            asset_fiat = - transaction_fiat
            asset_trade = transaction_fiat / price

            self.fiat = self.fiat + asset_fiat * (1 - trading_fees)
            self.asset = self.asset + asset_trade 
                
    def get_portfolio_distribution(self):
        return {
            "asset":self.asset,
            "fiat":self.fiat,
        }

class TargetPortfolio(Portfolio):
    def __init__(self, position ,value, price, init_value):
        super().__init__(
            asset = position * value / price,
            fiat = (1-position) * value,
            init_value = init_value
        )
