class Portfolio:
    def __init__(self, asset, fiat, interest_asset = 0, interest_fiat = 0):
        self.asset =asset
        self.fiat =fiat
        # 레버리지를 사용하면 interest가 붙는다.
        self.interest_asset = interest_asset
        self.interest_fiat = interest_fiat
        
    def valorisation(self, price):
        """
        레버리지 사용율(borrow interest)을 고려해서 포트폴리오의 가치를 나타낸다. 
        """
        return sum([
            self.asset * price,
            self.fiat,
            - self.interest_asset * price,
            - self.interest_fiat
        ])
        
    def real_position(self, price):
        return (self.asset - self.interest_asset) * price / self.valorisation(price)
    
    def position(self, price):
        return self.asset * price / self.valorisation(price)
    
    def trade_to_position(self, position, price, trading_fees):
        # Repay interest
        current_position = self.position(price)
        interest_reduction_ratio = 1
    
        # position의 방향은 유지하면서, position의 크기를 줄일 때
        # 레버리지 사용으로 인한 자산을 먼저 청산합니다. 
        # 그때, 레버리지 사용을 통한 borrowing fee를 빼줍니다. 
        # interest asset에 대해서 줄어드는 interest 값을 빼줍니다.
        if (position <= 0 and current_position < 0): # short position에서 short position 추가  
            interest_reduction_ratio = min(1, position/current_position)
        elif (position >= 1 and current_position > 1): # long position에서 long position 추가
            interest_reduction_ratio = min(1, (position-1)/(current_position-1))
            
        # interest_reduction_ratio가 1보다 크다는 의미는 leverage asset을 늘린다는 의미입니다.
        if interest_reduction_ratio < 1:
            self.asset = self.asset - (1-interest_reduction_ratio) * self.interest_asset
            self.fiat = self.fiat - (1-interest_reduction_ratio) * self.interest_fiat
            self.interest_asset = interest_reduction_ratio * self.interest_asset
            self.interest_fiat = interest_reduction_ratio * self.interest_fiat
        
        # Proceed to trade
        asset_trade = (position * self.valorisation(price) / price - self.asset)
        
        # long 거래
        if asset_trade > 0:
            asset_trade = asset_trade / (1 - trading_fees + trading_fees * position)
            asset_fiat = - asset_trade * price
            self.asset = self.asset + asset_trade * (1 - trading_fees)
            self.fiat = self.fiat + asset_fiat
            
        # shrot 거래
        else:
            asset_trade = asset_trade / (1 - trading_fees * position)
            asset_fiat = - asset_trade * price
            self.asset = self.asset + asset_trade 
            self.fiat = self.fiat + asset_fiat * (1 - trading_fees)
            
            
    def update_interest(self, borrow_interest_rate):
        # 공매도를 하면, self.asset이 음수가 됨. -> interest_asset이 적절하게 업데이트됨.
        self.interest_asset = max(0, -self.asset) * borrow_interest_rate
        # 공매수를 하면, self.fiat가 음수가 됨. -> interest_asset이 적절하게 업데이트됨.
        self.interest_fiat = max(0, -self.fiat) * borrow_interest_rate
    
    def __str__(self): return f"{self.__class__.__name__}({self.__dict__})"
    
    def describe(self, price): print("Value : ", self.valorisation(price), "Position : ", self.position(price))
    
    def get_portfolio_distribution(self):
        return {
            "asset":max(0, self.asset),
            "fiat":max(0, self.fiat),
            "borrowed_asset":max(0, -self.asset),
            "borrowed_fiat":max(0, -self.fiat),
            "interest_asset":self.interest_asset,
            "interest_fiat":self.interest_fiat,
        }

class TargetPortfolio(Portfolio):
    def __init__(self, position ,value, price):
        super().__init__(
            asset = position * value / price,
            fiat = (1-position) * value,
            interest_asset = 0,
            interest_fiat = 0
        )
