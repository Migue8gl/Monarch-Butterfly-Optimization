#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <set>
#include <iomanip>
#include <thread>
using namespace std;
using namespace std::chrono;

vector<vector<double>> data_matrix;
vector<char> class_vector;
random_device r;
// double Seed = r();
double Seed;

void readData(string file)
{
    vector<vector<string>> data_matrix_aux;
    string ifilename = file;
    ifstream ifile;
    istream *input = &ifile;

    ifile.open(ifilename.c_str());

    if (!ifile)
    {
        std::cerr << "[ERROR]Couldn't open the file" << endl;
        std::cerr << "[Ex.] Are you sure you are in the correct path?" << endl;
        std::exit(1);
    }

    string data;
    int cont = 0, cont_aux = 0;
    char aux;
    vector<string> aux_vector;
    bool finish = false;

    // Leo número de atributos y lo guardo en contador
    do
    {
        *input >> data;
        if (data == "@attribute")
            cont++;
    } while (data != "@data"); // A partir de aquí leemos datos

    data = "";

    // Mientras no lleguemos al final leemos datos
    while (!(*input).eof())
    {
        // Leemos caracter a caracter
        *input >> aux;

        /* Si hemos terminado una linea de datos la guardamos en la matrix de datos
        y reiniciamos el contador auxiliar (nos dice por qué dato vamos) */
        if (finish)
        {
            data_matrix_aux.push_back(aux_vector);
            aux_vector.clear();
            cont_aux = 0;
            finish = false;
        }

        /* Si hay una coma el dato ha terminado de leerse y lo almacenamos, en caso
        contrario seguimos leyendo caracteres y almacenandolos en data*/
        if (aux != ',' && cont_aux < cont)
        {
            data += aux;
            // Si hemos llegado al penultimo elemento hemos terminado
            if (cont_aux == cont - 1)
            {
                cont_aux++;
                aux_vector.push_back(data);
                data = "";
                finish = true;
            }
        }
        else
        {
            aux_vector.push_back(data);
            data = "";
            cont_aux++;
        }
    }

    vector<double> vect_aux;

    for (vector<vector<string>>::iterator it = data_matrix_aux.begin(); it != data_matrix_aux.end(); it++)
    {
        vect_aux.clear();
        for (vector<string>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            if (jt == it->end() - 1)
                class_vector.push_back((*jt)[0]);
            else
                vect_aux.push_back(stod(*jt));
        }
        data_matrix.push_back(vect_aux);
    }
}

void normalizeData(vector<vector<double>> &data)
{
    double item = 0.0;           // Característica individual
    double max_item = -999999.0; // Valor máximo del rango de valores
    double min_item = 999999.0;  // Valor minimo del rango de valores

    // Buscamos los máximos y mínimos
    for (vector<vector<double>>::iterator it = data.begin(); it != data.end(); it++)
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            item = *jt;

            if (item > max_item)
                max_item = item;

            if (item < min_item)
                min_item = item;
        }

    // Normalizamos aplicando x_iN = (x_i - min) / (max - min)
    for (vector<vector<double>>::iterator it = data.begin(); it != data.end(); it++)
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
            *jt = (*jt - min_item) / (max_item - min_item);
}

pair<vector<vector<vector<double>>>, vector<vector<char>>> createPartitions()
{
    vector<vector<double>> data_m_aux = data_matrix;
    vector<char> class_v_aux = class_vector;

    // Mezclo aleatoriamente la matriz original
    /*srand(Seed);
    random_shuffle(begin(data_m_aux), end(data_m_aux));
    srand(Seed);
    random_shuffle(begin(class_vector), end(class_vector));*/

    const int MATRIX_SIZE = data_matrix.size();
    vector<vector<double>>::iterator it = data_m_aux.begin();
    vector<char>::iterator jt = class_v_aux.begin();

    // Particiones y puntero que las irá recorriendolas para insertar datos
    vector<vector<double>> g1, g2, g3, g4, g5, *g_aux;
    vector<char> g1c, g2c, g3c, g4c, g5c, *g_aux2;
    int cont = 0, cont_grupos = 0;
    bool salir = false;

    // Mientras no se hayan insertado todos los datos en todos los grupos
    while (cont != MATRIX_SIZE && cont_grupos < 5)
    {
        // Elegimos la partición que toque
        switch (cont_grupos)
        {
        case 0:
            g_aux = &g1;
            g_aux2 = &g1c;
            break;
        case 1:
            g_aux = &g2;
            g_aux2 = &g2c;
            break;
        case 2:
            g_aux = &g3;
            g_aux2 = &g3c;
            break;
        case 3:
            g_aux = &g4;
            g_aux2 = &g4c;
            break;
        case 4:
            g_aux = &g5;
            g_aux2 = &g5c;
            break;
        }

        // Vamos rellenando la partición pertinente
        for (int k = 0; k < MATRIX_SIZE / 5 && !salir; k++)
        {
            g_aux->push_back(*it);
            g_aux2->push_back(*jt);
            it++;
            jt++;
            cont++;

            /* Si estamos en el último grupo y quedan todavía elementos, seguir
            insertándolos en este último */
            if (cont_grupos == 4)
            {
                if (it != data_m_aux.end())
                    k--;
                else
                    salir = true;
            }
        }
        cont_grupos++;
    }
    vector<vector<vector<double>>> d = {g1, g2, g3, g4, g5};
    vector<vector<char>> c = {g1c, g2c, g3c, g4c, g5c};
    pair<vector<vector<vector<double>>>, vector<vector<char>>> partitions = make_pair(d, c);

    return partitions;
}

char KNN_Classifier(vector<vector<double>> &data, vector<vector<double>>::iterator &elem, vector<char> &elemClass, vector<double> &w)
{
    vector<double> distancia;
    vector<char> clases;
    vector<char>::iterator cl = elemClass.begin();
    vector<double>::iterator wi = w.begin();
    vector<double>::iterator ej;
    double sumatoria = 0;
    double dist_e = 0;

    for (vector<vector<double>>::iterator e = data.begin(); e != data.end(); e++)
    {
        // Si el elemento es él mismo no calculamos distancia, pues es 0
        if (elem != e)
        {
            sumatoria = 0;
            ej = elem->begin();
            wi = w.begin();

            // Calculamos distancia de nuestro elemento con el resto
            for (vector<double>::iterator ei = e->begin(); ei != e->end(); ei++)
            {
                sumatoria += *wi * pow(*ej - *ei, 2);
                ej++;
                wi++;
            }
            dist_e = sqrt(sumatoria);
            distancia.push_back(dist_e);
            clases.push_back(*cl);
        }
        cl++;
    }

    vector<double>::iterator it;
    vector<char>::iterator cl_dist_min = clases.begin();

    double distMin = 99999;
    char vecinoMasProxClass;

    // Nos quedamos con el que tenga minima distancia, es decir, su vecino más próximo
    for (it = distancia.begin(); it != distancia.end(); it++)
    {
        if (*it < distMin)
        {
            distMin = *it;
            vecinoMasProxClass = *cl_dist_min;
        }
        cl_dist_min++;
    }

    return vecinoMasProxClass;
}

double calculaAciertos(vector<vector<double>> &muestras, vector<char> &clases, vector<double> &w)
{
    double instBienClasificadas = 0.0;
    double numIntanciasTotal = float(muestras.size());
    char cl_1NN;
    vector<char>::iterator c_it = clases.begin();

    for (vector<vector<double>>::iterator it = muestras.begin(); it != muestras.end(); it++)
    {
        cl_1NN = KNN_Classifier(muestras, it, clases, w);

        if (cl_1NN == *c_it)
            instBienClasificadas += 1.0;
        c_it++;
    }

    return instBienClasificadas / numIntanciasTotal;
}

void execute(pair<vector<vector<vector<double>>>, vector<vector<char>>> &part, vector<double> (*alg)(vector<vector<double>> &, vector<char> &, bool), bool bl)
{
    vector<double> w;
    vector<vector<vector<double>>>::iterator data_test = part.first.begin();
    vector<vector<char>>::iterator class_test = part.second.begin();
    vector<vector<double>> aux_data_fold;
    vector<char> aux_class_fold;
    vector<vector<vector<double>>>::iterator it;
    vector<vector<char>>::iterator jt;

    double tasa_clas = 0;
    double tasa_red = 0;
    double agregado = 0;
    double alpha = 0.5;
    unsigned int cont_red = 0;
    double TS_media = 0, TR_media = 0, A_media = 0;
    int cont = 0;

    auto momentoInicio = high_resolution_clock::now();

    // Iteramos 5 veces ejecutando el algoritmo
    while (cont < 5)
    {
        jt = part.second.begin();
        aux_data_fold.clear();
        aux_class_fold.clear();
        cont_red = 0;

        // Creamos particiones train
        for (it = part.first.begin(); it != part.first.end(); it++)
        {
            // Si es una partición test no la añadimos a training
            if (it != data_test && jt != class_test)
            {
                aux_data_fold.insert(aux_data_fold.end(), (*it).begin(), (*it).end());
                aux_class_fold.insert(aux_class_fold.end(), (*jt).begin(), (*jt).end());
            }
            jt++;
        }

        // Ejecución del algoritmo
        auto partInicio = high_resolution_clock::now();
        w = alg(aux_data_fold, aux_class_fold, bl);
        auto partFin = high_resolution_clock::now();

        cont_red = 0;
        for (vector<double>::iterator wi = w.begin(); wi != w.end(); wi++)
        {
            if (*wi < 0.1)
            {
                cont_red += 1;
                *wi = 0.0;
            }
        }

        tasa_clas = calculaAciertos(*data_test, *class_test, w);
        tasa_red = float(cont_red) / float(w.size());
        agregado = alpha * tasa_clas + (1 - alpha) * tasa_red;

        milliseconds tiempo_part = duration_cast<std::chrono::milliseconds>(partFin - partInicio);

        std::cout << "[PART " << cont + 1 << "] | Tasa_clas: " << tasa_clas << endl;
        std::cout << "[PART " << cont + 1 << "] | Tasa_red: " << tasa_red << endl;
        std::cout << "[PART " << cont + 1 << "] | Fitness: " << agregado << endl;
        std::cout << "[PART " << cont + 1 << "] | Tiempo_ejecucion: " << tiempo_part.count() << " ms\n\n";
        std::cout << "-------------------------------------------\n"
                  << endl;

        TS_media += tasa_clas;
        TR_media += tasa_red;
        A_media += agregado;

        cont++;
        data_test++;
        class_test++;
    }
    auto momentoFin = high_resolution_clock::now();

    milliseconds tiempo = duration_cast<std::chrono::milliseconds>(momentoFin - momentoInicio);

    std::cout << "***** (RESULTADOS FINALES) *****\n"
              << endl;
    std::cout << "Tasa_clas_media: " << TS_media / 5.0 << endl;
    std::cout << "Tasa_red_media: " << TR_media / 5.0 << endl;
    std::cout << "Fitness_medio: " << A_media / 5.0 << endl;
    std::cout << "Tiempo_ejecucion_medio: " << tiempo.count() << " ms";
}

double evalua(vector<vector<double>> &muestra, vector<char> &muestra_clases, vector<double> &poblacion)
{
    double tasa_clas = 0;
    double tasa_red = 0;
    double cont_reducc = 0;
    double f;

    for (vector<double>::iterator it = poblacion.begin(); it != poblacion.end(); it++)
    {
        if (*it < 0.1)
        {
            cont_reducc += 1.0;
            *it = 0;
        }
    }
    tasa_clas = calculaAciertos(muestra, muestra_clases, poblacion);
    tasa_red = float(cont_reducc) / float(poblacion.size());
    f = tasa_red * 0.5 + tasa_clas * 0.5;

    return f;
}

vector<double> evalua_pob(vector<vector<double>> &muestra, vector<char> &muestra_clases, vector<vector<double>> &poblacion)
{
    double tasa_clas;
    double tasa_red;
    unsigned int cont_reducc;
    vector<double> f;
    f.resize(poblacion.size());
    int i = 0;

    for (vector<vector<double>>::iterator it = poblacion.begin(); it != poblacion.end(); it++)
    {
        cont_reducc = 0;
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            if (*jt < 0.1)
            {
                cont_reducc += 1;
                *jt = 0;
            }
        }
        tasa_clas = calculaAciertos(muestra, muestra_clases, *it);
        tasa_red = float(cont_reducc) / float((*it).size());
        f[i] = tasa_red * 0.5 + tasa_clas * 0.5;
        i++;
    }

    return f;
}

double busquedaLocal(vector<vector<double>> &muestra, vector<char> &clase, vector<double> &w)
{
    const int maxIter = 1000;
    int cont = 0;
    double varianza = 0.3, alpha = 0.5;

    // Creo vector z y un generador de distribución normal
    vector<double> z(w.size());
    vector<double>::iterator z_it;
    normal_distribution<double> normal_dist(0.0, sqrt(varianza));
    double s = r();
    mt19937 other_eng(Seed);
    auto genNormalDist = [&normal_dist, &other_eng]()
    {
        return normal_dist(other_eng);
    };

    double fun_objetivo = 0;
    double max_fun = -99999.0;
    double w_aux;

    fun_objetivo = evalua(muestra, clase, w);
    cont++;
    max_fun = fun_objetivo;

    // Mientras no se superen las iteraciones máximas o los vecinos permitidos
    while (cont < maxIter)
    {
        generate(begin(z), end(z), genNormalDist);
        z_it = z.begin();

        for (vector<double>::iterator it = w.begin(); it != w.end(); it++)
        {
            // Guardamos w original
            w_aux = *it;

            // Mutación normal
            *it += *z_it;

            if (*it < 0)
                *it = 0;
            else if (*it > 1)
                *it = 1;

            fun_objetivo = evalua(muestra, clase, w);
            cont++;

            // Si hemos mejorado el umbral a mejorar cambia, vamos maximizando la función
            if (fun_objetivo > max_fun)
                max_fun = fun_objetivo;
            else // Si no hemos mejorado nos quedamos con la w anterior
                *it = w_aux;
            z_it++;
        }
    }
    return max_fun;
}

void migration(vector<vector<double>> &subpob1, vector<vector<double>> &subpob2, double periodo, double p)
{
    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    double n;
    int index;
    int k_index = 0;
    int c = 0;

    for (vector<vector<double>>::iterator it = subpob1.begin(); it != subpob1.end(); it++)
    {
        k_index = 0;
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            n = gen() * periodo;

            c++;

            if (n <= p)
            {
                index = r() % subpob1.size();
                (*it)[k_index] = subpob1[index][k_index];
            }
            else
            {
                index = r() % subpob2.size();
                (*it)[k_index] = subpob2[index][k_index];
            }
            k_index++;
        }
    }
}

double levyFlight()
{
    uniform_real_distribution<double> dist(0.0, 0.01);
    mt19937 eng(r());

    return tan(M_PI * dist(eng));
}

void adjust(vector<vector<double>> &subpob1, vector<vector<double>> &subpob2, vector<int> indexbest1, vector<int> indexbest2, double p, vector<double> evals, double BAR, double alpha)
{
    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    double n, dx;
    int index;
    int i = 0;
    int k_index = 0;

    for (vector<vector<double>>::iterator it = subpob2.begin(); it != subpob2.end(); it++)
    {
        k_index = 0;
        i = 0;
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            n = gen();

            if (n <= p)
            {
                if (evals[indexbest1[i]] > evals[indexbest2[i] + subpob1.size()])
                {
                    (*it)[k_index] = subpob1[indexbest1[i]][k_index];
                    i++;
                }
                else
                {
                    (*it)[k_index] = subpob2[indexbest2[i]][k_index];
                    i++;
                }
            }
            else
            {
                index = r() % subpob2.size();
                (*it)[k_index] = subpob2[index][k_index];

                if (n > BAR)
                {
                    dx = levyFlight();
                    (*it)[k_index] = (*it)[k_index] + alpha * (dx - 0.5);
                }
            }
            k_index++;
        }
    }
}

vector<double> algMBO(vector<vector<double>> &muestra, vector<char> &clase, bool BL)
{
    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    // Inizializo población de mariposas NP
    vector<vector<double>> np(muestra.size());
    vector<double> w(muestra.begin()->size());
    for (int i = 0; i < np.size(); i++)
    {
        generate(begin(w), end(w), gen);
        np[i] = w;
    }

    int t = 1;
    const int maxGen = 10;
    double BAR = 5.0 / 12.0;
    double p = 5.0 / 12.0;
    double peri = 1.2;
    double smax = 1.0;
    double alpha = 0;
    int np1_size = ceil(p * np.size());
    int np2_size = np.size() - np1_size;
    double cont = 0;
    vector<double> evals;
    vector<vector<double>> np2, np1;
    vector<int> indices1(np1_size), indices2(np2_size);
    std::iota(indices1.begin(), indices1.end(), 0);
    std::iota(indices2.begin(), indices2.end(), 0);
    evals = evalua_pob(muestra, clase, np);

    while (t < maxGen)
    {
        // Ordenamos en orden de mejor a peor fitness
        sort(indices1.begin(), indices1.end(),
             [&](double A, double B) -> bool
             {
                 return evals[A] > evals[B];
             });

        sort(indices2.begin(), indices2.end(),
             [&](double A, double B) -> bool
             {
                 return evals[A] > evals[B];
             });

        np1.clear();
        np2.clear();

        for (vector<vector<double>>::iterator it = np.begin(); it != np.end(); it++)
        {
            if (cont < np1_size)
                np1.push_back(*it);
            else
                np2.push_back(*it);

            cont++;
        }

        alpha = smax / pow(t, 2);

        thread mig_thread(migration, std::ref(np1), std::ref(np2), peri, p);
        thread aju_thread(adjust, std::ref(np1), std::ref(np2), indices1, indices2, p, evals, BAR, alpha);
        
        mig_thread.join();
        aju_thread.join();

        np1.insert(np1.end(), np2.begin(), np2.end());
        np.swap(np1);
        std::shuffle(np.begin(), np.end(), default_random_engine(r()));
        evals = evalua_pob(muestra, clase, np);
        t++;
        cont = 0;
    }

    double best_eval = *evals.begin();
    int index = 0, best_solution_index = 0;
    for (vector<double>::iterator it = evals.begin(); it != evals.end(); it++)
    {
        if (*it > best_eval)
        {
            best_eval = *it;
            best_solution_index = index;
        }
        index++;
    }

    if (BL)
        busquedaLocal(muestra, clase, np[best_solution_index]);

    return np[best_solution_index];
}

int main(int nargs, char *args[])
{
    char *arg[4];
    string option;
    string path;

    if (nargs <= 2)
    {
        std::cerr << "[ERROR] Wrong execution pattern" << endl;
        std::cerr << "[Ex.] ./main {seed} [1-3] " << endl;
        std::cerr << "[Pd:] 1=spectf-heart, 2=parkinsons, 3=ionosphere" << endl;
    }
    Seed = atof(args[1]);
    option = args[2];

    if (option == "1")
        path = "./bin/spectf-heart.arff";
    else if (option == "2")
        path = "./bin/parkinsons.arff";
    else if (option == "3")
        path = "./bin/ionosphere.arff";
    else
    {
        std::cerr << "[ERROR] Parámetro no reconocido..." << endl;
        std::cerr << "[Ex.] Tienes que definir que data-set: 1-spectf-heart, 2-parkinsons, 3-ionosphere..." << endl;
        std::cerr << "[Ex.] ./main {seed} [1-3] " << endl;
        std::exit(1);
    }

    readData(path);
    normalizeData(data_matrix);

    pair<vector<vector<vector<double>>>, vector<vector<char>>> part;
    part = createPartitions();

    std::srand(Seed);
    std::cout << "\nSemilla: " << setprecision(10) << Seed << endl;

    std::cout << "\n------------(ALGORITMO DE OPTIMIZACIÓN DE MARIPOSAS MONARCAS)------------\n\n";
    execute(part, algMBO, false);

    std::cout << "\n\n------------(ALGORITMO DE OPTIMIZACIÓN DE MARIPOSAS MONARCAS [BUSQUEDA_LOCAL])------------\n\n";
    execute(part, algMBO, true);

    std::cout << endl
              << endl;
}
