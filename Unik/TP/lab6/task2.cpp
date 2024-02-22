
#include <iostream>
#include <string>
#include <memory>

class Rocket {
public:
	Rocket(const std::string& name, float height, float diameter, float mass)
		: m_name(name), m_height(height), m_diameter(diameter), m_mass(mass)
	{}

	~Rocket() {}

	void getInfo() const {
		std::cout << "Rocket " << m_name << " specifications:" << std::endl;
		std::cout << "Height: " << m_height << " meters" << std::endl;
		std::cout << "Diameter: " << m_diameter << " meters" << std::endl;
		std::cout << "Mass: " << m_mass << " kg" << std::endl;
	}

	void launch() {
		std::cout << "Rocket " << m_name << " is launching..." << std::endl;
	}

private:
	std::string m_name;
	float m_height;
	float m_diameter;
	float m_mass;
};

int main() {
	try {
		std::unique_ptr<Rocket> rocketPtr = std::make_unique<Rocket>("Falcon 9", 70.0, 3.7, 549054.0);
		rocketPtr->getInfo();
		rocketPtr->launch();

		// Здесь происходит генерация исключения
		throw std::runtime_error("An unexpected error occurred");

		// Эта строка не будет выполнена из-за исключения
		rocketPtr->launch();
	} catch (const std::exception& e) {
		std::cerr << e.what() << '\n';
	}

	return 0;
}

