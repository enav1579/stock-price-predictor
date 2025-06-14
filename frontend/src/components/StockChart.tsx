import { useQuery } from 'react-query';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { Paper, Typography, Box, CircularProgress } from '@mui/material';
import axios from 'axios';

interface StockChartProps {
  ticker: string;
}

const StockChart = ({ ticker }: StockChartProps) => {
  const { data, isLoading, error } = useQuery(
    ['historical', ticker],
    async () => {
      const response = await axios.get(`http://localhost:8000/stocks/historical/${ticker}`);
      return response.data;
    }
  );

  if (isLoading) {
    return (
      <Paper sx={{ p: 2, height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <CircularProgress />
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper sx={{ p: 2, height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography color="error">Error loading chart data</Typography>
      </Paper>
    );
  }

  if (!data?.data) {
    return (
      <Paper sx={{ p: 2, height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography>No data available</Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2, height: 400 }}>
      <Typography variant="h6" gutterBottom>
        Price History
      </Typography>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data.data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            tick={{ fill: '#fff' }}
            tickLine={{ stroke: '#fff' }}
          />
          <YAxis 
            tick={{ fill: '#fff' }}
            tickLine={{ stroke: '#fff' }}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#1e1e1e',
              border: '1px solid #333',
              color: '#fff'
            }}
          />
          <Line 
            type="monotone" 
            dataKey="close" 
            stroke="#90caf9" 
            dot={false}
            name="Close Price"
          />
        </LineChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export default StockChart; 