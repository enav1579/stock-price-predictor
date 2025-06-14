import { useState } from 'react';
import { 
  Box, 
  Container, 
  Grid, 
  Paper, 
  TextField, 
  Button, 
  Typography,
  ThemeProvider,
  createTheme,
  CssBaseline
} from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import StockChart from './components/StockChart';
import StockMetrics from './components/StockMetrics';
import PredictionCard from './components/PredictionCard';
import TechnicalIndicators from './components/TechnicalIndicators';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
});

const queryClient = new QueryClient();

function App() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [inputValue, setInputValue] = useState('');

  const handleAddTicker = () => {
    if (inputValue && !tickers.includes(inputValue.toUpperCase())) {
      setTickers([...tickers, inputValue.toUpperCase()]);
      setInputValue('');
    }
  };

  const handleRemoveTicker = (ticker: string) => {
    setTickers(tickers.filter(t => t !== ticker));
  };

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <Box sx={{ flexGrow: 1, minHeight: '100vh', py: 3 }}>
          <Container maxWidth="xl">
            <Grid container spacing={3}>
              {/* Header */}
              <Grid item xs={12}>
                <Paper sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography variant="h4" component="h1" sx={{ flexGrow: 1 }}>
                    Stock Market Dashboard
                  </Typography>
                  <TextField
                    label="Add Stock Ticker"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleAddTicker()}
                    size="small"
                  />
                  <Button 
                    variant="contained" 
                    onClick={handleAddTicker}
                    disabled={!inputValue}
                  >
                    Add
                  </Button>
                </Paper>
              </Grid>

              {/* Stock List */}
              <Grid item xs={12}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Watchlist
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {tickers.map((ticker) => (
                      <Paper
                        key={ticker}
                        sx={{
                          p: 1,
                          display: 'flex',
                          alignItems: 'center',
                          gap: 1,
                          bgcolor: 'background.paper',
                        }}
                      >
                        <Typography>{ticker}</Typography>
                        <Button
                          size="small"
                          color="error"
                          onClick={() => handleRemoveTicker(ticker)}
                        >
                          ×
                        </Button>
                      </Paper>
                    ))}
                  </Box>
                </Paper>
              </Grid>

              {/* Main Content */}
              {tickers.map((ticker) => (
                <Grid item xs={12} key={ticker}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h5" gutterBottom>
                      {ticker}
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={8}>
                        <StockChart ticker={ticker} />
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Grid container spacing={2}>
                          <Grid item xs={12}>
                            <StockMetrics ticker={ticker} />
                          </Grid>
                          <Grid item xs={12}>
                            <PredictionCard ticker={ticker} />
                          </Grid>
                          <Grid item xs={12}>
                            <TechnicalIndicators ticker={ticker} />
                          </Grid>
                        </Grid>
                      </Grid>
                    </Grid>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Container>
        </Box>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App; 