
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

typedef float fl_t;

using namespace::std;

template <class T>
ostream &operator <<(ostream &out, const vector<T> &v) {
  for (const auto &i : v) {
    out << i << ' ';
  }
  out << '\n';
  return out;
}

static auto _solve(const vector<fl_t> &a, const vector<fl_t> &b, const vector<fl_t> &c, const vector<fl_t> &d)
{
        int n = a.size();

        vector<fl_t> cc(c);
        vector<fl_t> dc(d);

        cc.at(0) /= b.at(0);
        dc.at(0) /= b.at(0);

        n -= 1;
        for (int i = 1; i < n; ++i) {
            cc.at(i) /= b.at(i) - a.at(i) * cc.at(i - 1);
            dc.at(i) = (dc.at(i) - a.at(i) * dc.at(i - 1)) / (b.at(i) - a.at(i) * cc.at(i - 1));
        }

        dc.at(n) = (dc.at(n) - a.at(n) * dc.at(n - 1)) / (b.at(n) - a.at(n) * cc.at(n - 1));

        for (int i = n - 1; i >= 0; --i) {
            dc.at(i) -= cc.at(i) * dc.at(i + 1);
        }

        return dc;
}

static void solve(int n, fl_t dt, fl_t dx, const vector<fl_t> &tube, vector<vector<fl_t>> &temp)
{
        vector<fl_t> A(n, 0);
        vector<fl_t> B(n, 0);
        vector<fl_t> C(n, 0);

        for (int i = 0; i < n; ++i) {
                fl_t r1, r2;
                if (i == 0) {
                  r1 = 0;
                } else {
                  r1 = tube.at(i - 1) + tube.at(i);
                }

                if (i == n - 1) {
                  r2 = 0;
                } else {
                  r2 = tube.at(i) + tube.at(i + 1);
                }
          
                A.at(i) = -(dt * r1) / (2 * dx * dx);
                B.at(i) = 1 + (dt * (r1 + r2)) / (2 * dx * dx);
                C.at(i) = -(dt * r2) / (2 * dx * dx);
        }

        B.front() = 1;
        C.front() = 0;
        B.back() = 1;
        A.back() = 0;
        A.front() = 0;
        C.back() = 0;

        for (int i = 1; i < temp.size(); ++i) {
                temp.at(i) = _solve(A, B, C, temp.at(i - 1));
        }
}

fl_t err(int time, int n, const vector<vector<fl_t>> &out, const vector<vector<fl_t>> &temp)
{
        fl_t err = 0;

        for (int t = 0; t < time; ++t) {
                for (int i = 0; i < n; ++i) {
                        if (temp.at(t).at(i) > 0) {
                                fl_t e = temp.at(t).at(i) - out.at(t).at(i);
                                err += e * e;
                        }
                }
        }

        return err;
}

void backward(int maxiter, int t, int n, fl_t dt, fl_t dx, const vector<vector<fl_t>> &temp) {
  vector<fl_t> tube(n);
  vector<fl_t> best_tube(n);
  fl_t best_err = 1e9;
  vector<vector<fl_t>> out(temp);
  
  srandom(time(nullptr));
  
  for (int it = 0; it < maxiter; ++it) {
    if (it % 1000 == 0) {
      printf("it %d\n", it);
    }
    for (int i = 0; i < n; ++i) {
      tube[i] = static_cast<fl_t>(random()) / RAND_MAX;
      tube[i] = tube[i] * 0.2 + 0.1;
    }

    solve(n, dt, dx, tube, out);
    fl_t e = err(t, n, out, temp);

    if (e < best_err) {
      printf("iter %d: err %.8f\n", it, e);
      //cout << tube;
      best_tube.swap(tube);
      best_err = e;
    }
  }
}

int main()
{
        int n, time;
        fl_t dt, dx;

        cin >> n >> time >> dt >> dx;

        vector<fl_t> tube(n);
        for (int i = 0; i < n; ++i) {
                cin >> tube.at(i);
        }

        vector<vector<fl_t>> temp(time, vector<fl_t>(n));

        for (int t = 0; t < time; ++t) {
          for (int i = 0; i < n; ++i) {
            cin >> temp.at(t).at(i);
          }
        }

        printf("n %d\n", n);
        printf("time %d\n", time);
        printf("dt %f\n", dt);
        printf("dx %f\n", dx);

        //cout << "tube\n" << tube << '\n';
        //cout << "temp\n" << temp << '\n';

        vector<vector<fl_t>> out(temp);
        solve(n, dt, dx, tube, out);

        cout << "solved temp\n" << out << '\n';
        printf("ERROR: %.8f\n", err(time, n, out, temp));

        backward(100000, time, n, dt, dx, temp);
}
