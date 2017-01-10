/*
  Test BaggedDataset
 */

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "BaggedDataset.h"

const size_t InstanceLabelDim = 3;
const size_t BagLabelDim = 2;


class BaggedDatasetTest : public ::testing::Test {
public:
  typedef BaggedDataset< 1, 1 > SimpleBaggedDatasetType;
  typedef BaggedDataset< BagLabelDim, InstanceLabelDim > BaggedDatasetType;
  typedef BaggedDatasetType::MatrixType MatrixType;
  typedef BaggedDatasetType::IndexVectorType IndexVectorType;
  typedef BaggedDatasetType::BagLabelVectorType BagLabelVectorType;
  typedef BaggedDatasetType::InstanceLabelVectorType InstanceLabelVectorType;
  
protected:
  virtual void SetUp() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> disBags(10, 200);
    std::uniform_int_distribution<size_t> disBagSize(10, 100);
    std::uniform_int_distribution<size_t> disDimension(25, 2000);

    numberOfBags = disBags( gen );
    bagSize = disBagSize( gen );
    dimension = disDimension( gen );
    numberOfInstances = numberOfBags * bagSize;

    instances = MatrixType::Random( numberOfInstances, dimension );
    bagMembership = IndexVectorType( numberOfInstances );
    for ( size_t i = 0, index = 0; i < static_cast<size_t>(bagMembership.size()); ++i ) {
      if ( i >= (index + 1) * bagSize ) {
	++index;
      }
      bagMembership(i) = index;
    }
    bagLabels = BagLabelVectorType::Random( numberOfBags, BagLabelDim );
    instanceLabels = InstanceLabelVectorType::Zero( numberOfInstances, InstanceLabelDim );
    bags = BaggedDatasetType( instances, bagMembership, bagLabels, instanceLabels );
  }


  BaggedDatasetType bags;
  MatrixType instances;
  BagLabelVectorType bagLabels;
  InstanceLabelVectorType instanceLabels;
  IndexVectorType bagMembership;
  size_t numberOfBags, bagSize, dimension, numberOfInstances;
};

/*
*/
TEST_F( BaggedDatasetTest, NumberOfBags ) {
  ASSERT_EQ( numberOfBags, bags.NumberOfBags() );
}

TEST_F( BaggedDatasetTest, NumberOfInstances ) {
  ASSERT_EQ( numberOfInstances, bags.NumberOfInstances() );
}

TEST_F( BaggedDatasetTest, Dimension ) {
  ASSERT_EQ( dimension, bags.Dimension() );
}

TEST_F( BaggedDatasetTest, Indices ) {
    ASSERT_EQ( bagMembership, bags.Indices() );
    ASSERT_EQ(  numberOfBags - 1, bags.Indices()(numberOfInstances - 1) );
}

TEST_F( BaggedDatasetTest, Instances ) {
  ASSERT_EQ( instances, bags.Instances() );
}

TEST_F( BaggedDatasetTest, BagLabels ) {
  ASSERT_EQ( bagLabels, bags.BagLabels() );
}

TEST_F( BaggedDatasetTest, InstanceLabels ) {
  ASSERT_EQ( instanceLabels, bags.InstanceLabels() );
  ASSERT_EQ( 0, bags.InstanceLabels().sum() );
}


TEST_F( BaggedDatasetTest, WrongNumberOfInstanceLabels ) {
  InstanceLabelVectorType tooFew( numberOfInstances - 1, InstanceLabelDim );
  InstanceLabelVectorType tooMany( numberOfInstances + 1, InstanceLabelDim );
  ASSERT_THROW( BaggedDatasetType( instances, bagMembership, bagLabels, tooFew ), std::logic_error );
  ASSERT_THROW( BaggedDatasetType( instances, bagMembership, bagLabels, tooMany ), std::logic_error );
  ASSERT_THROW( bags.InstanceLabels( tooFew ), std::logic_error );
  ASSERT_THROW( bags.InstanceLabels( tooMany ), std::logic_error );
}

TEST_F( BaggedDatasetTest, BadBagMembershipIndex ) {
  IndexVectorType badIndex = bagMembership;
  badIndex( numberOfBags - 1 ) = numberOfBags; // Off by one
  ASSERT_THROW( BaggedDatasetType( instances, badIndex, bagLabels, instanceLabels ), std::logic_error );
}


TEST_F( BaggedDatasetTest, Equal ) {
  ASSERT_EQ( bags, bags );
  auto bags2 = BaggedDatasetType( instances, bagMembership, bagLabels, instanceLabels );
  ASSERT_EQ( bags, bags2 );
}

TEST_F( BaggedDatasetTest, CopyCtor ) {
  auto bags2 = bags;
  ASSERT_EQ( bags, bags2 );
}

TEST_F( BaggedDatasetTest, NotEqual ) {
  auto bags2 = BaggedDatasetType::Random(numberOfBags, bagSize, dimension );
  ASSERT_NE( bags, bags2 );
}


TEST_F( BaggedDatasetTest, LoadSave ) {  
  std::string path = "BaggedDatasetTest.LoadSave.bags";
  std::ofstream os( path );
  bags.Save( os );

  std::ifstream is( path );
  BaggedDatasetType bags2 = BaggedDatasetType::Load( is );

  ASSERT_EQ( bags, bags2 );
}

TEST_F( BaggedDatasetTest, LoadTextSmall ) {
  typedef BaggedDataset< 1, 1 > SimpleBaggedDatasetType;
  std::string path = "data/Small.bags";
  std::ifstream is(path);
  SimpleBaggedDatasetType smallBags = SimpleBaggedDatasetType::LoadText( is );

  ASSERT_EQ(4, smallBags.NumberOfBags());
  ASSERT_EQ(4, smallBags.NumberOfInstances());
  ASSERT_EQ(4, smallBags.Dimension());

  double n = 1;
  for (size_t i = 0; i < smallBags.NumberOfInstances(); ++i ) {
    for (size_t j = 0; j < smallBags.Dimension(); ++j ) {
      ASSERT_NEAR(n++, smallBags.Instances()(i,j), 1e-6);
    }
  }

}

TEST_F( BaggedDatasetTest, LoadSaveLoadTextSmall ) {
  std::string path = "data/Small.bags"; 
  std::ifstream is(path);
  SimpleBaggedDatasetType smallBags = SimpleBaggedDatasetType::LoadText( is );

  std::string path2 = "data/Small.saved.bags"; 
  std::ofstream os(path2);
  smallBags.SaveText( os );

  std::ifstream is2(path2);
  SimpleBaggedDatasetType smallBags2 = SimpleBaggedDatasetType::LoadText( is2, true );

  ASSERT_EQ(smallBags, smallBags2);
}


TEST_F( BaggedDatasetTest, LoadSaveLoadTextBig ) {  
  std::string orgpath = "data/D1-meanmidpoints.test.bags.csv";
  std::ifstream orgis( orgpath );
  SimpleBaggedDatasetType orgbags = SimpleBaggedDatasetType::LoadText( orgis, true );

  std::string path = "data/big.saved.bags";
  std::ofstream os( path );
  orgbags.SaveText( os );

  std::ifstream is( path );
  SimpleBaggedDatasetType bags2 = SimpleBaggedDatasetType::LoadText( is, true );

  ASSERT_EQ( orgbags, bags2 );
}



TEST_F( BaggedDatasetTest, Join ) {
  size_t N2 = numberOfBags/2;
  size_t N3 = numberOfBags - N2;
  BaggedDatasetType bags2( instances.topRows( N2*bagSize ),
			   bagMembership.topRows( N2*bagSize ),
			   bagLabels.topRows( N2 ),
			   instanceLabels.topRows( N2*bagSize ) );
  BaggedDatasetType bags3( instances.bottomRows( N3*bagSize ),
			   bagMembership.topRows( N3*bagSize ),
			   bagLabels.bottomRows( N3 ),
			   instanceLabels.bottomRows( N3*bagSize ) );
  
  auto bags4 = BaggedDatasetType::Join( bags2, bags3 );

  ASSERT_EQ( bags, bags4 );  
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
